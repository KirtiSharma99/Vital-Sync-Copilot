/*
  VitalSync BED_01 - Model3 live test (experimental)
  Hardware/pins retained from Model2 sketch:
    MAX30102 + MPU6050 on I2C SDA=21, SCL=22; MPU6050 addr=0x68
    DS18B20 on GPIO4 (read for display only; not a Model3 input)

  Model3 classifier: 150-tree Random Forest, 44 features, classes:
    0 GOOD, 1 ACCEPTABLE, 2 POOR

  Intended pipeline from the Python post-filter builder:
    100 Hz -> 0.3 Hz causal highpass -> 4-ref NLMS (8 taps, mu=.05, eps=1e-6)
    -> causal 3-point moving average -> 8 s window, 2 s hop -> 44 features -> RF

  IMPORTANT LIMITATIONS:
    * Python builder normalized IMU reference channels using the RMS over each
      COMPLETE recording. Live hardware cannot know future recording RMS, so this
      sketch estimates reference RMS from a 20-second startup calibration period.
      This is an approximation, not exact feature parity.
    * PTT dataset labels are provisional weak-rule candidate labels, not human-
      verified SQI ground truth. Treat predictions as experimental only.
    * Sensor timing is paired one MPU read per MAX30102 FIFO sample; verify actual
      effective sample rate and alignment on hardware.
    * Verify feature parity with Python on captured ESP32 samples before relying
      on the reported offline score. This is NOT a medical device.
*/

#include <Arduino.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <math.h>
#include <float.h>
#include "MAX30105.h"
#include "VitalSync_Model3_Types.h"
#include "VitalSync_Model3_RF_ESP32.h"
#include "VitalSync_Model2_RF_ESP32.h"
#include "ContactModel.h"
#include <ArduinoJson.h>
#include <Preferences.h>

// ---------- Existing VitalSync wiring ----------
#define PIN_SDA       21
#define PIN_SCL       22
#define PIN_DS18B20    4
#define MPU_ADDR     0x68

// ---------- Acquisition and model settings ----------
static const float FS_HZ = 100.0f;
static const uint32_t WINDOW_N = 800;       // 8 seconds at 100 Hz
static const uint32_t HOP_N = 200;          // prediction every 2 seconds
static const uint32_t CAL_N = 2000;         // 20-second startup reference-RMS calibration
static const float ACC_LSB_PER_G = 8192.0f; // MPU6050 +/-4 g
static const float GYR_LSB_PER_DPS = 65.5f; // MPU6050 +/-500 dps
static const float G_TO_MS2 = 9.80665f;
static const float DPS_TO_RAD = 0.017453292519943295f;
// Sensor runs in the ORIGINAL Model 3 configuration (LED 0x24, ADC 16384, avg 4 @ 400 Hz -> 100 Hz),
// so Model 3/2 see exactly the signal they were developed on (no rescaling).
// Model 1 (contact) was trained on counts from LED 0x1F @ ADC 4096, avg 1 @ 100 Hz. Counts scale with
// LED current / ADC range, so only the Model 1 input is converted: collection = old * 4 * (31/36) ~= 3.44.
// Model 1 is a coarse 3-threshold tree, so it tolerates a small error here. Tune CONTACT_COUNT_SCALE if needed.
static const uint8_t LED_CURRENT = 0x24;
static const float M3_SIGNAL_SCALE = 1.0f;
static const float CONTACT_COUNT_SCALE = 4.0f*(31.0f/36.0f);
static const bool STREAM_FEATURES = true;
static const bool STREAM_RAW = false;
static const uint8_t NLMS_TAPS = 8;
static const uint8_t NLMS_REFS = 4;
static const float NLMS_MU = 0.05f;
static const float NLMS_EPS = 1.0e-6f;
static const float REF_CLIP = 8.0f;
static const uint16_t MAF_N = 3;
static const float PI_F = 3.14159265358979323846f;

MAX30105 ppg;
OneWire oneWire(PIN_DS18B20);
DallasTemperature ds(&oneWire);

struct RawSample {
  float red, ir;
  float ax, ay, az;     // m/s^2
  float gx, gy, gz;     // rad/s, bias corrected
};

struct HighPass03 {
  // scipy.signal.butter(2, 0.3, 'highpass', fs=100, output='sos')
  // scipy sos row: [b0,b1,b2,1,a1,a2]
  static constexpr float b0 = 0.98675978f;
  static constexpr float b1 = -1.97351956f;
  static constexpr float b2 = 0.98675978f;
  static constexpr float a1 = -1.97334425f;
  static constexpr float a2 = 0.97369487f;
  float z1 = 0.0f, z2 = 0.0f;
  bool initialized = false;
  void reset() { z1=0; z2=0; initialized=false; }
  float process(float x) {
    // Match scipy sosfilt_zi(sos) * x[0] steady-state initialization.
    if (!initialized) {
      z1 = -0.98675978f * x;
      z2 =  0.98675978f * x;
      initialized = true;
    }
    float y = b0*x + z1;
    z1 = b1*x - a1*y + z2;
    z2 = b2*x - a2*y;
    return y;
  }
};

static RawSample *calBuf = nullptr;  // temporary 20-second heap buffer; freed after calibration
static RawSample rawRing[WINDOW_N];
static float redHpRing[WINDOW_N], irHpRing[WINDOW_N];
static float redPostRing[WINDOW_N], irPostRing[WINDOW_N];
static uint32_t ringWrite = 0, validSamples = 0, sincePrediction = 0;
static bool calibrated = false;
static uint32_t calCount = 0, sampleCounter = 0;
static float gyroBiasX=0, gyroBiasY=0, gyroBiasZ=0;
static float refRms[NLMS_REFS] = {1,1,1,1};
static float nlmsWRed[NLMS_REFS][NLMS_TAPS] = {{0}};
static float nlmsWIr[NLMS_REFS][NLMS_TAPS] = {{0}};
static float nlmsHistRed[NLMS_REFS][NLMS_TAPS] = {{0}};
static float nlmsHistIr[NLMS_REFS][NLMS_TAPS] = {{0}};
static uint8_t nlmsPosRed = 0, nlmsPosIr = 0;
static float mafRed[2] = {0,0}, mafIr[2] = {0,0};
static HighPass03 hpAccX, hpAccY, hpAccZ, hpGyro, hpRed, hpIr;
static float tempC = NAN;
static uint32_t lastTempMs = 0;
static uint32_t packetSequence = 0, windowId = 0;
static ContactPrediction lastContact = {255, "UNKNOWN", 0, 0, 0, 0};
static VitalSyncModel3Prediction lastM3 = {2,{0,0,1}};
static VitalSyncRFResult lastM2 = {0,{1,0,0}};   // Model 2 (motion corruption): 0 CLEAN, 1 MILD, 2 SEVERE
// If true, SEVERE motion corruption from Model 2 also withholds the HR value.
static const bool M2_GATES_HR = false;

// ---------- Robust SQI + fast contact settings ----------
static const uint32_t CONTACT_HOP_N       = 25;      // contact re-evaluated every 0.25 s (was every 2 s)
static const uint8_t  CONTACT_CONFIRM     = 2;       // consecutive checks needed to change contact state
static const float    CONTACT_FAST_IR_MAX = 37868.0f;// same floor as Model 1 tree: recent IR max below => NO_CONTACT at once
static const uint32_t HEARTBEAT_N         = 50;      // publish at least every 0.5 s
static const uint32_t SQI_SETTLE_SAMPLES  = 600;     // 6 s of continuous good contact before SQI/HR are trusted
static const float    SQI_ALPHA           = 0.6f;    // weight of newest window in probability smoothing
static const uint8_t  SQI_CONFIRM_RUNS    = 2;       // windows needed to enter/leave POOR (hysteresis)
static const float    HR_MIN_BPM          = 30.0f, HR_MAX_BPM = 220.0f;
static const float    HR_DISAGREE_SOFT    = 15.0f;
static const float    CLIP_MAX_FRAC       = 0.05f;

static float    sqiProb[3] = {0,0,1};
static uint8_t  sqiClass = 2;                 // stabilized Model 3 class (0 GOOD, 1 ACCEPTABLE, 2 POOR)
static uint8_t  sqiPoorRun = 0, sqiGoodRun = 0;
static const char *sqiReason = "WARMING_UP";
static bool     sqiWindowValid = false;
static uint32_t contactGoodRun = 0, sinceContactCheck = 0, lastPublishSample = 0;
static uint8_t  contactPendingKlass = 255, contactPendingCount = 0;
static float    heldHr = NAN;
static float    spo2Smooth = NAN, spo2Out = NAN;   // SpO2 estimate (smoothed) / value published this cycle
static bool     m2Valid = false;                  // Model 2 has scored a clean, in-contact window
// SpO2 (ratio-of-ratios, Maxim-style quadratic). UNCALIBRATED estimate: tune A/B/C against a reference oximeter.
static const float SPO2_A = -45.060f, SPO2_B = 30.354f, SPO2_C = 94.845f, SPO2_ALPHA = 0.4f;
static float spo2Offset = 0.0f;           // one-point calibration offset (stored in flash); set with CAL_SPO2,<ref>
static float spo2PI = NAN;                // perfusion index (%) of the latest window
static float spo2Hist[5]; static uint8_t spo2HistN = 0, spo2HistPos = 0;
static float hrTrack = NAN, hrCand = NAN; static uint8_t hrCandCount = 0, hrMissCount = 0;
static Preferences prefs;
static float lastReliableHr=NAN, lastReliableAt=0;

static void publishTelemetry(const char *state, float hr) {
  // One newline-delimited JSON record for the USB Serial dashboard.
  StaticJsonDocument<1536> d;
  d["bed_id"]="bed-01";d["node_id"]="VS_NODE_01";d["firmware_version"]="serial-prototype";
  d["window_id"]=++windowId;d["sequence"]=++packetSequence;d["timestamp_ms"]=millis();d["sampling_rate_hz"]=100;d["expected_samples"]=WINDOW_N;d["received_samples"]=validSamples;
  d["contact_model"]=lastContact.name; d["ir_mean"]=lastContact.irMean; d["red_mean"]=lastContact.redMean; d["ir_max"]=lastContact.irMax; d["red_min"]=lastContact.redMin; d["model2"]=(lastContact.klass==1)?"NO_SIGNAL":(!m2Valid?"SETTLING":vitalSyncModel2ClassName(lastM2.predictedClass)); JsonArray p2=d.createNestedArray("model2_probabilities");for(int i=0;i<3;i++)p2.add((lastContact.klass==1||!m2Valid)?0.0f:lastM2.probabilities[i]); d["model3"]=(lastContact.klass==1)?"NO_SIGNAL":(!sqiWindowValid?"SETTLING":vitalSyncModel3ClassName(sqiClass)); d["settle_pct"]=(int)min(100UL,(unsigned long)(contactGoodRun*100UL/SQI_SETTLE_SAMPLES)); d["model3_raw"]=vitalSyncModel3ClassName(lastM3.predictedClass); d["sqi_reason"]=sqiReason;
  JsonArray p=d.createNestedArray("model3_probabilities");for(int i=0;i<3;i++)p.add((lastContact.klass==1||!sqiWindowValid)?0.0f:sqiProb[i]);
  // Arduino C++ cannot form a ternary expression from float and nullptr.
  // Assign missing values explicitly so the serial JSON record contains `null`.
  if (isfinite(hr)) d["hr_bpm"] = hr; else d["hr_bpm"] = nullptr;
  if (isfinite(spo2Out)) d["spo2_estimate"] = spo2Out; else d["spo2_estimate"] = nullptr;
  d["spo2_offset"] = spo2Offset; if (isfinite(spo2PI)) d["perfusion_index"] = spo2PI;
  if (isfinite(tempC)) d["temperature_c"] = tempC; else d["temperature_c"] = nullptr;
  d["signal_state"]=state;
  if(isfinite(lastReliableHr)){d["last_reliable_hr_bpm"]=lastReliableHr;d["last_reliable_timestamp_ms"]=lastReliableAt;}
  Serial.print("VS_JSON,");
  serializeJson(d,Serial);
  Serial.println();
}

// Window scratch arrays. Static to avoid placing large buffers on the loop stack.
static float wRedRaw[WINDOW_N], wIrRaw[WINDOW_N];
static float wRedHp[WINDOW_N], wIrHp[WINDOW_N];
static float wRed[WINDOW_N], wIr[WINDOW_N];
static float wAccMag[WINDOW_N], wGyroMag[WINDOW_N];
static float templateRed[20][64], templateIr[20][64];

static inline float sqf(float x) { return x*x; }
static inline float clampf(float x, float lo, float hi) { return x<lo?lo:(x>hi?hi:x); }

// ---------- MPU6050 low-level access ----------
static bool mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR); Wire.write(reg); Wire.write(val);
  return Wire.endTransmission() == 0;
}
static bool mpuInit() {
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x75);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MPU_ADDR,(uint8_t)1) != 1) return false;
  uint8_t who=Wire.read();
  Serial.printf("# MPU6050 WHO_AM_I=0x%02X\n",who);
  bool ok=true;
  ok &= mpuWrite(0x6B,0x01); delay(50);
  ok &= mpuWrite(0x19,9);      // 100 Hz
  ok &= mpuWrite(0x1A,0x03);   // DLPF
  ok &= mpuWrite(0x1B,0x08);   // +/-500 dps
  ok &= mpuWrite(0x1C,0x08);   // +/-4 g
  return ok;
}
static inline int16_t readS16(const uint8_t *b,int i) {
  return (int16_t)(((uint16_t)b[i]<<8)|b[i+1]);
}
static float readGyroDpsAxis(uint8_t axis);

static bool mpuRead(RawSample &s) {
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x3B);
  if (Wire.endTransmission(false)!=0) return false;
  if (Wire.requestFrom((uint8_t)MPU_ADDR,(uint8_t)14)!=14) return false;
  uint8_t b[14]; for(int i=0;i<14;i++) b[i]=Wire.read();
  s.ax=(readS16(b,0)/ACC_LSB_PER_G)*G_TO_MS2;
  s.ay=(readS16(b,2)/ACC_LSB_PER_G)*G_TO_MS2;
  s.az=(readS16(b,4)/ACC_LSB_PER_G)*G_TO_MS2;
  s.gx=(readS16(b,8)/GYR_LSB_PER_DPS-gyroBiasX)*DPS_TO_RAD;
  s.gy=(readS16(b,10)/GYR_LSB_PER_DPS-gyroBiasY)*DPS_TO_RAD;
  s.gz=(readS16(b,12)/GYR_LSB_PER_DPS-gyroBiasZ)*DPS_TO_RAD;
  return true;
}
static bool calibrateGyro() {
  const int N=500; double sx=0,sy=0,sz=0;
  Serial.println("# Keep MPU6050 still for 5 seconds: gyro bias calibration");
  for(int i=0;i<N;i++) {
    RawSample s;
    if(!mpuRead(s)) return false;
    sx += readGyroDpsAxis(0); sy += readGyroDpsAxis(1); sz += readGyroDpsAxis(2);
    delay(10);
  }
  gyroBiasX=(float)(sx/N); gyroBiasY=(float)(sy/N); gyroBiasZ=(float)(sz/N);
  Serial.printf("# gyro_bias_dps=%.4f,%.4f,%.4f\n",gyroBiasX,gyroBiasY,gyroBiasZ);
  return true;
}
// Raw gyro read for calibration in degrees/second. Called only during setup.
static float readGyroDpsAxis(uint8_t axis) {
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x43 + axis*2);
  if(Wire.endTransmission(false)!=0) return 0;
  if(Wire.requestFrom((uint8_t)MPU_ADDR,(uint8_t)2)!=2) return 0;
  uint8_t hi=Wire.read(),lo=Wire.read();
  int16_t v=(int16_t)(((uint16_t)hi<<8)|lo);
  return (float)v/GYR_LSB_PER_DPS;
}

static void serviceTemp() {
  uint32_t now=millis();
  if(now-lastTempMs>5000) {
    ds.requestTemperatures();
    tempC=ds.getTempCByIndex(0);
    if(tempC==DEVICE_DISCONNECTED_C || tempC < -50 || tempC > 125) tempC=NAN;
    lastTempMs=now;
  }
}

// ---------- Reference normalization calibration ----------
static void estimateReferenceRms() {
  HighPass03 hx,hy,hz,hg;
  double sum[NLMS_REFS]={0,0,0,0};
  double sum2[NLMS_REFS]={0,0,0,0};
  for(uint32_t i=0;i<CAL_N;i++) {
    const RawSample &s=calBuf[i];
    float gm=sqrtf(s.gx*s.gx+s.gy*s.gy+s.gz*s.gz);
    float v[4]={hx.process(s.ax),hy.process(s.ay),hz.process(s.az),hg.process(gm)};
    for(int c=0;c<4;c++){sum[c]+=v[c];sum2[c]+=(double)v[c]*v[c];}
  }
  for(int c=0;c<4;c++) {
    double mean=sum[c]/CAL_N;
    double variance=sum2[c]/CAL_N-mean*mean;
    refRms[c]=(float)sqrt(variance>1e-12?variance:1e-12);
    Serial.printf("# ref_rms[%d]=%.6f\n",c,refRms[c]);
  }
}

static float nlmsStep(float target,const float refs[NLMS_REFS],
                      float weights[NLMS_REFS][NLMS_TAPS],
                      float history[NLMS_REFS][NLMS_TAPS], uint8_t &pos) {
  for(int c=0;c<NLMS_REFS;c++) history[c][pos]=refs[c];
  double yhat=0.0, norm=NLMS_EPS;
  for(int c=0;c<NLMS_REFS;c++) for(int t=0;t<NLMS_TAPS;t++) {
    int p=(int)pos-t; while(p<0)p+=NLMS_TAPS;
    float x=history[c][p]; yhat += weights[c][t]*x; norm += (double)x*x;
  }
  float e=target-(float)yhat;
  float gain=NLMS_MU*e/(float)norm;
  for(int c=0;c<NLMS_REFS;c++) for(int t=0;t<NLMS_TAPS;t++) {
    int p=(int)pos-t; while(p<0)p+=NLMS_TAPS;
    weights[c][t] += gain*history[c][p];
  }
  pos=(pos+1)%NLMS_TAPS;
  return e;
}

static void resetStreamingFilters() {
  hpAccX.reset(); hpAccY.reset(); hpAccZ.reset(); hpGyro.reset(); hpRed.reset(); hpIr.reset();
  memset(nlmsWRed,0,sizeof(nlmsWRed)); memset(nlmsWIr,0,sizeof(nlmsWIr));
  memset(nlmsHistRed,0,sizeof(nlmsHistRed));
  memset(nlmsHistIr,0,sizeof(nlmsHistIr));
  nlmsPosRed=0; nlmsPosIr=0;
  mafRed[0]=mafRed[1]=mafIr[0]=mafIr[1]=0;
}

static void processOneSample(const RawSample &s,bool store) {
  float refs[4];
  float gyroMag=sqrtf(s.gx*s.gx+s.gy*s.gy+s.gz*s.gz);
  refs[0]=clampf(hpAccX.process(s.ax)/refRms[0],-REF_CLIP,REF_CLIP);
  refs[1]=clampf(hpAccY.process(s.ay)/refRms[1],-REF_CLIP,REF_CLIP);
  refs[2]=clampf(hpAccZ.process(s.az)/refRms[2],-REF_CLIP,REF_CLIP);
  refs[3]=clampf(hpGyro.process(gyroMag)/refRms[3],-REF_CLIP,REF_CLIP);
  float rHp=hpRed.process(s.red*M3_SIGNAL_SCALE);
  float iHp=hpIr.process(s.ir*M3_SIGNAL_SCALE);
  float rResidual=nlmsStep(rHp,refs,nlmsWRed,nlmsHistRed,nlmsPosRed);
  float iResidual=nlmsStep(iHp,refs,nlmsWIr,nlmsHistIr,nlmsPosIr);
  float rPost=(rResidual+mafRed[0]+mafRed[1])/3.0f;
  mafRed[1]=mafRed[0]; mafRed[0]=rResidual;
  float iPost=(iResidual+mafIr[0]+mafIr[1])/3.0f;
  mafIr[1]=mafIr[0]; mafIr[0]=iResidual;
  if(store) {
    rawRing[ringWrite]=s;
    redHpRing[ringWrite]=rHp; irHpRing[ringWrite]=iHp;
    redPostRing[ringWrite]=rPost; irPostRing[ringWrite]=iPost;
    ringWrite=(ringWrite+1)%WINDOW_N;
    if(validSamples<WINDOW_N)validSamples++;
  }
}

static void finishCalibration() {
  Serial.println("# Estimating IMU reference scales from 20-second startup sample...");
  estimateReferenceRms();
  resetStreamingFilters();
  ringWrite=0; validSamples=0;
  Serial.println("# Replaying calibration buffer through highpass + NLMS + MAF...");
  for(uint32_t i=0;i<CAL_N;i++) processOneSample(calBuf[i],true);
  free(calBuf); calBuf=nullptr;
  sincePrediction=0;
  calibrated=true;
  Serial.println("READY,MODEL3_CALIBRATED,first prediction in about 2 seconds");
}

// ---------- Statistics helpers ----------
static void meanStd(const float *x,int n,float &m,float &sd) {
  double s=0; for(int i=0;i<n;i++)s+=x[i]; m=(float)(s/n);
  double v=0; for(int i=0;i<n;i++)v+=sqf(x[i]-m); sd=sqrtf((float)(v/n));
}
static float corrSigned(const float *a,const float *b,int n) {
  float ma,sa,mb,sb; meanStd(a,n,ma,sa); meanStd(b,n,mb,sb);
  if(sa<1e-9f||sb<1e-9f)return 0;
  double c=0; for(int i=0;i<n;i++)c+=(a[i]-ma)*(b[i]-mb);
  return (float)(c/(n*sa*sb));
}
static float medianOf(float *a,int n) {
  if(n<=0)return NAN;
  for(int i=1;i<n;i++){float v=a[i];int j=i-1;while(j>=0&&a[j]>v){a[j+1]=a[j];j--;}a[j+1]=v;}
  return (n&1)?a[n/2]:0.5f*(a[n/2-1]+a[n/2]);
}

// ---------- Peak detection with prominence and minimum-distance rule ----------
static int detectPeaks(const float *x,int n,float polarity,int *out,float *outProm) {
  const int MAXC=150, MIN_DIST=30;
  int cand[MAXC],nc=0; float height[MAXC],prom[MAXC]; bool keep[MAXC];
  float m,sd; meanStd(x,n,m,sd);
  float mn=FLT_MAX,mx=-FLT_MAX;
  for(int i=0;i<n;i++){float v=polarity*x[i];if(v<mn)mn=v;if(v>mx)mx=v;}
  float threshold=fmaxf(0.35f*sd,0.08f*(mx-mn));
  if(threshold<1e-8f)threshold=1e-8f;
  for(int i=1;i<n-1&&nc<MAXC;i++) {
    float v=polarity*x[i];
    if(v>polarity*x[i-1] && v>=polarity*x[i+1]) {cand[nc]=i;height[nc]=v;nc++;}
  }
  for(int i=0;i<nc;i++)keep[i]=true;
  // scipy find_peaks distance selection: retain taller peaks when closer than 30 samples.
  for(int a=0;a<nc;a++) if(keep[a]) {
    for(int b=a+1;b<nc;b++) if(keep[b] && abs(cand[a]-cand[b])<MIN_DIST) {
      if(height[a]>=height[b])keep[b]=false;else {keep[a]=false;break;}
    }
  }
  int count=0;
  for(int k=0;k<nc;k++) if(keep[k]) {
    int p=cand[k]; float leftMin=polarity*x[p],rightMin=leftMin;
    for(int j=p-1;j>=0;j--){float v=polarity*x[j];if(v>height[k])break;if(v<leftMin)leftMin=v;}
    for(int j=p+1;j<n;j++){float v=polarity*x[j];if(v>height[k])break;if(v<rightMin)rightMin=v;}
    float pr=height[k]-fmaxf(leftMin,rightMin);
    if(pr>=threshold && count<MAXC){out[count]=p;outProm[count]=pr;count++;}
  }
  // Candidate scan is time-ordered; keep in chronological order.
  return count;
}

static BeatResult beatMetrics(const float *x) {
  BeatResult best; best.hr=NAN; best.ibiCv=NAN; best.ibiRmssd=NAN; best.templateCorr=NAN;
  best.promMean=NAN; best.promCv=NAN; best.peakCount=0;
  float bestCv=1e6f;
  for(int poln=0;poln<2;poln++) {
    float pol=poln==0?1.0f:-1.0f;
    int p[150];float pr[150];int n=detectPeaks(x,WINDOW_N,pol,p,pr);
    if(n>40)n=40;
    float ibi[39];int ni=n-1;
    for(int i=0;i<ni;i++)ibi[i]=(p[i+1]-p[i])/FS_HZ;
    float cv=NAN,hr=NAN,rmssd=NAN;
    if(ni>0) {
      float im,is;meanStd(ibi,ni,im,is);if(im>0)cv=is/im;
      float tmp[39];for(int i=0;i<ni;i++)tmp[i]=ibi[i];float med=medianOf(tmp,ni);if(med>0)hr=60.0f/med;
      if(ni>2){double z=0;for(int i=1;i<ni;i++)z+=sqf(ibi[i]-ibi[i-1]);rmssd=sqrtf((float)(z/(ni-1)));}
    }
    float rankCv=isfinite(cv)?cv:1e6f;
    if(rankCv<bestCv || (fabsf(rankCv-bestCv)<1e-7f && n>best.peakCount)) {
      bestCv=rankCv;best.hr=hr;best.ibiCv=cv;best.ibiRmssd=rmssd;best.peakCount=n;
      for(int i=0;i<n;i++)best.peaks[i]=p[i];
      double ps=0,ps2=0;for(int i=0;i<n;i++){ps+=pr[i];ps2+=(double)pr[i]*pr[i];}
      best.promMean=n?(float)(ps/n):NAN;
      if(n && ps>0){double vv=ps2/n-(ps/n)*(ps/n);best.promCv=(float)(sqrt(vv>0?vv:0)/(ps/n));}
      else best.promCv=NAN;
    }
  }
  // Pairwise normalized beat-template correlations (up to 20 templates).
  int nt=0;
  for(int b=0;b<best.peakCount-1 && nt<20;b++) {
    int a=best.peaks[b],z=best.peaks[b+1];
    if(z-a<30||z-a>200)continue;
    float mm[64];
    for(int j=0;j<64;j++) {
      float pos=a+(z-a)*(float)j/63.0f;int k=(int)pos;float frac=pos-k;
      mm[j]=x[k]*(1-frac)+x[(k+1<WINDOW_N?k+1:k)]*frac;
    }
    float m,sd;meanStd(mm,64,m,sd);if(sd<1e-8f)continue;
    for(int j=0;j<64;j++) {
      float v=(mm[j]-m)/sd;
      if(x==wIr) templateIr[nt][j]=v;
      else templateRed[nt][j]=v;
    }
    nt++;
  }
  if(nt>=2) {
    float cs[190];int nc=0;
    float (*tb)[64]=(x==wIr)?templateIr:templateRed;
    for(int a=0;a<nt;a++)for(int b=a+1;b<nt;b++)cs[nc++]=corrSigned(tb[a],tb[b],64);
    best.templateCorr=medianOf(cs,nc);
  }
  return best;
}

// Welch PSD equivalent: Hann window, 512 points, 50% overlap, constant detrend.
static SpectralResult spectralMetrics(const float *x) {
  const int N=512, H=256;
  double accum[26]={0};
  double winEnergy=0;
  float hann[N];
  for(int n=0;n<N;n++){hann[n]=0.5f-0.5f*cosf(2.0f*PI_F*n/N);winEnergy+=(double)hann[n]*hann[n];}
  for(int seg=0;seg<2;seg++) {
    int start=seg*H; double m=0;for(int n=0;n<N;n++)m+=x[start+n];m/=N;
    for(int k=0;k<=25;k++) {
      double re=0,im=0;
      for(int n=0;n<N;n++) {
        double a=2.0*PI_F*k*n/N;
        double v=(x[start+n]-(float)m)*hann[n];
        re+=v*cos(a); im-=v*sin(a);
      }
      double p=(re*re+im*im)/(FS_HZ*winEnergy);
      if(k>0 && k<N/2)p*=2.0;
      accum[k]+=p*0.5;
    }
  }
  double total=0,maxP=-1;int maxK=3;int count=0;
  for(int k=3;k<=25;k++){total+=accum[k];count++;if(accum[k]>maxP){maxP=accum[k];maxK=k;}}
  SpectralResult r;
  if(total<=1e-20){r.dom=NAN;r.peakRatio=0;r.entropy=1;r.bandPower=0;return r;}
  double ent=0;for(int k=3;k<=25;k++){double p=accum[k]/total;if(p>1e-15)ent-=p*log(p);}
  r.dom=(float)maxK*FS_HZ/N;r.peakRatio=(float)(maxP/total);
  r.entropy=(count>1)?(float)(ent/log((double)count)):0;
  r.bandPower=(float)total;
  return r;
}

static void loadWindow() {
  for(uint32_t i=0;i<WINDOW_N;i++) {
    uint32_t p=(ringWrite+i)%WINDOW_N;
    const RawSample &s=rawRing[p];
    wRedRaw[i]=s.red; wIrRaw[i]=s.ir;
    wRedHp[i]=redHpRing[p]; wIrHp[i]=irHpRing[p];
    wRed[i]=redPostRing[p]; wIr[i]=irPostRing[p];
  }
  float axm=0,aym=0,azm=0;
  for(int i=0;i<WINDOW_N;i++){const RawSample &s=rawRing[(ringWrite+i)%WINDOW_N];axm+=s.ax;aym+=s.ay;azm+=s.az;}
  axm/=WINDOW_N;aym/=WINDOW_N;azm/=WINDOW_N;
  for(int i=0;i<WINDOW_N;i++) {
    const RawSample &s=rawRing[(ringWrite+i)%WINDOW_N];
    float dx=s.ax-axm,dy=s.ay-aym,dz=s.az-azm;
    wAccMag[i]=sqrtf(dx*dx+dy*dy+dz*dz);
    wGyroMag[i]=sqrtf(s.gx*s.gx+s.gy*s.gy+s.gz*s.gz);
  }
}

static bool extractFeatures(float f[VITALSYNC_M3_N_FEATURES]) {
  loadWindow();
  for(int i=0;i<VITALSYNC_M3_N_FEATURES;i++)f[i]=NAN;
  BeatResult br=beatMetrics(wRed), bi=beatMetrics(wIr);
  SpectralResult sr=spectralMetrics(wRed), si=spectralMetrics(wIr);
  float redMean,redSd,irMean,irSd;
  meanStd(wRedRaw,WINDOW_N,redMean,redSd);meanStd(wIrRaw,WINDOW_N,irMean,irSd);
  redMean*=M3_SIGNAL_SCALE; irMean*=M3_SIGNAL_SCALE;   // match the scale of the highpassed PPG
  double redPostSq=0,irPostSq=0;
  for(int i=0;i<WINDOW_N;i++){redPostSq+=(double)wRed[i]*wRed[i];irPostSq+=(double)wIr[i]*wIr[i];}
  float redPostRms=sqrtf((float)(redPostSq/WINDOW_N));
  float irPostRms=sqrtf((float)(irPostSq/WINDOW_N));
  float accMean,accSd,gyroMean,gyroSd;
  meanStd(wAccMag,WINDOW_N,accMean,accSd);meanStd(wGyroMag,WINDOW_N,gyroMean,gyroSd);
  double accSq=0,redHpSq=0,irHpSq=0;float accPeak=0;
  int clipR=0,clipI=0,flatR=0,flatI=0;
  for(int i=0;i<WINDOW_N;i++) {
    accSq+=sqf(wAccMag[i]);if(wAccMag[i]>accPeak)accPeak=wAccMag[i];
    redHpSq+=sqf(wRedHp[i]);irHpSq+=sqf(wIrHp[i]);
    if(wRedRaw[i]<=1.0f||wRedRaw[i]>=262140.0f)clipR++;
    if(wIrRaw[i]<=1.0f||wIrRaw[i]>=262140.0f)clipI++;
    if(i>0){if(fabsf(wRedRaw[i]-wRedRaw[i-1])<1e-6f)flatR++;if(fabsf(wIrRaw[i]-wIrRaw[i-1])<1e-6f)flatI++;}
  }
  // Exact header order: 0..43.
  f[0]=br.hr; f[1]=bi.hr;
  if(isfinite(br.hr)&&isfinite(bi.hr))f[2]=fabsf(br.hr-bi.hr);
  f[3]=(float)br.peakCount; f[4]=(float)bi.peakCount;
  f[5]=br.ibiCv; f[6]=bi.ibiCv;
  f[7]=br.ibiRmssd; f[8]=bi.ibiRmssd;
  f[9]=br.templateCorr; f[10]=bi.templateCorr;
  f[11]=br.promMean; f[12]=bi.promMean;
  f[13]=br.promCv; f[14]=bi.promCv;
  f[15]=sr.dom; f[16]=si.dom;
  f[17]=sr.peakRatio; f[18]=si.peakRatio;
  f[19]=sr.entropy; f[20]=si.entropy;
  f[21]=redPostRms; f[22]=irPostRms;
  float rmin=wRed[0],rmax=wRed[0],imin=wIr[0],imax=wIr[0];
  for(int i=1;i<WINDOW_N;i++){if(wRed[i]<rmin)rmin=wRed[i];if(wRed[i]>rmax)rmax=wRed[i];if(wIr[i]<imin)imin=wIr[i];if(wIr[i]>imax)imax=wIr[i];}
  f[23]=rmax-rmin; f[24]=imax-imin;
  f[25]=redMean; f[26]=irMean;
  f[27]=redPostRms/fmaxf(fabsf(redMean),1e-9f);
  f[28]=irPostRms/fmaxf(fabsf(irMean),1e-9f);
  f[29]=(float)clipR/WINDOW_N; f[30]=(float)clipI/WINDOW_N;
  f[31]=(float)flatR/(WINDOW_N-1); f[32]=(float)flatI/(WINDOW_N-1);
  f[33]=sqrtf((float)(accSq/WINDOW_N)); f[34]=accPeak;
  double gyroSq=0;for(int i=0;i<WINDOW_N;i++)gyroSq+=sqf(wGyroMag[i]);
  f[35]=sqrtf((float)(gyroSq/WINDOW_N));
  f[36]=fabsf(corrSigned(wRed,wAccMag,WINDOW_N));
  f[37]=fabsf(corrSigned(wIr,wAccMag,WINDOW_N));
  f[38]=fabsf(corrSigned(wRed,wGyroMag,WINDOW_N));
  f[39]=fabsf(corrSigned(wIr,wGyroMag,WINDOW_N));
  float redIn=sqrtf((float)(redHpSq/WINDOW_N));
  float irIn=sqrtf((float)(irHpSq/WINDOW_N));
  f[40]=redPostRms/fmaxf(redIn,1e-9f);
  f[41]=irPostRms/fmaxf(irIn,1e-9f);
  f[42]=sr.bandPower; f[43]=si.bandPower;
  for(int i=0;i<VITALSYNC_M3_N_FEATURES;i++)if(!isfinite(f[i]))return false;
  return true;
}

// ---------- Model 2 (motion corruption) feature extraction ----------
// 34 features in the exact order of RF_FEATURE_NAMES in VitalSync_Model2_RF_ESP32.h.
// PPG input = 0.3 Hz highpassed raw red/IR (no NLMS). Acc in g, gyro in deg/s.
// NOTE: the original Python feature code was not supplied, so definitions are inferred from
// the feature names (see comments). Verify against Python on captured data.
static void skewKurt(const float *x,int n,float &sk,float &ku) {
  double m=0; for(int i=0;i<n;i++)m+=x[i]; m/=n;
  double m2=0,m3=0,m4=0;
  for(int i=0;i<n;i++){double d=x[i]-m;m2+=d*d;m3+=d*d*d;m4+=d*d*d*d;}
  m2/=n;m3/=n;m4/=n;
  if(m2<1e-12){sk=0;ku=0;return;}
  sk=(float)(m3/pow(m2,1.5)); ku=(float)(m4/(m2*m2)-3.0);   // excess kurtosis (scipy default)
}
static float absDiff(float a,float b){return (isfinite(a)&&isfinite(b))?fabsf(a-b):NAN;}

static void extractModel2Features(float f[VITALSYNC_RF_N_FEATURES]) {
  // wRedHp, wIrHp, wAccMag, wGyroMag, wRedRaw, wIrRaw are already loaded by extractFeatures().
  BeatResult br=beatMetrics(wRedHp), bi=beatMetrics(wIrHp);
  SpectralResult sr=spectralMetrics(wRedHp), si=spectralMetrics(wIrHp);
  SpectralResult sa=spectralMetrics(wAccMag), sg=spectralMetrics(wGyroMag);

  double accSq=0; float accPeak=0;
  double gSum=0,gSq=0;
  for(int i=0;i<WINDOW_N;i++){
    accSq+=sqf(wAccMag[i]); if(wAccMag[i]>accPeak)accPeak=wAccMag[i];
    float g=wGyroMag[i]/DPS_TO_RAD; gSum+=g; gSq+=(double)g*g;
  }
  float gMean=(float)(gSum/WINDOW_N);
  float gyroRmsDps=sqrtf((float)(gSq/WINDOW_N));
  float gyroStdDps=sqrtf(fmaxf(0.0f,(float)(gSq/WINDOW_N)-gMean*gMean));

  float rMean,rSd,iMean,iSd,rHpMean,rHpSd,iHpMean,iHpSd;
  meanStd(wRedRaw,WINDOW_N,rMean,rSd); meanStd(wIrRaw,WINDOW_N,iMean,iSd);
  rMean*=M3_SIGNAL_SCALE; iMean*=M3_SIGNAL_SCALE;
  meanStd(wRedHp,WINDOW_N,rHpMean,rHpSd); meanStd(wIrHp,WINDOW_N,iHpMean,iHpSd);
  float acdcR=rHpSd/fmaxf(fabsf(rMean),1e-9f), acdcI=iHpSd/fmaxf(fabsf(iMean),1e-9f);

  float rSk,rKu,iSk,iKu; skewKurt(wRedHp,WINDOW_N,rSk,rKu); skewKurt(wIrHp,WINDOW_N,iSk,iKu);

  f[0]=sqrtf((float)(accSq/WINDOW_N))/G_TO_MS2;
  f[1]=accPeak/G_TO_MS2;
  f[2]=sa.dom;
  f[3]=gyroRmsDps; f[4]=gyroStdDps; f[5]=sg.dom;
  f[6]=br.hr; f[7]=bi.hr; f[8]=absDiff(br.hr,bi.hr);
  f[9]=0.5f*(acdcR+acdcI);
  f[10]=absDiff(sr.dom,sa.dom); f[11]=absDiff(si.dom,sa.dom);
  f[12]=absDiff(sr.dom,sg.dom); f[13]=absDiff(si.dom,sg.dom);
  f[14]=br.ibiCv; f[15]=(float)br.peakCount; f[16]=sr.peakRatio; f[17]=sr.entropy;
  f[18]=br.templateCorr; f[19]=rSk; f[20]=rKu;
  f[21]=fabsf(corrSigned(wRedHp,wAccMag,WINDOW_N));
  f[22]=fabsf(corrSigned(wRedHp,wGyroMag,WINDOW_N));
  f[23]=sr.dom;
  f[24]=bi.ibiCv; f[25]=(float)bi.peakCount; f[26]=si.peakRatio; f[27]=si.entropy;
  f[28]=bi.templateCorr; f[29]=iSk; f[30]=iKu;
  f[31]=fabsf(corrSigned(wIrHp,wAccMag,WINDOW_N));
  f[32]=fabsf(corrSigned(wIrHp,wGyroMag,WINDOW_N));
  f[33]=si.dom;
  // Model 2 has no imputer; replace missing values with 0 so tree comparisons are defined.
  for(int i=0;i<VITALSYNC_RF_N_FEATURES;i++) if(!isfinite(f[i])) f[i]=0.0f;
}

// ---------- State machine + publishing ----------
static const char *currentState(bool &hrOk, bool &reliable) {
  hrOk=false; reliable=false;
  if(lastContact.klass==1) return "SENSOR_NOT_CONTACTED";
  if(lastContact.klass!=0) return "CONTACT_UNSTABLE";
  if(contactGoodRun<SQI_SETTLE_SAMPLES || !sqiWindowValid) return "SIGNAL_QUALITY_SETTLING";
  if(M2_GATES_HR && lastM2.predictedClass==2) return "READING_WITHHELD_MOTION";
  const bool hrPlausible=isfinite(heldHr)&&heldHr>=HR_MIN_BPM&&heldHr<=HR_MAX_BPM;
  if(sqiClass==VITALSYNC_M3_POOR) {
    // Keep updating HR/SpO2 but flag it: the number is shown with a low-quality warning.
    if(hrPlausible){hrOk=true; return "READING_LOW_QUALITY";}
    return "READING_WITHHELD_SIGNAL_QUALITY";
  }
  hrOk=hrPlausible; reliable=hrOk;
  return "READING_VALID";
}
static void publishNow() {
  bool ok,rel; const char *st=currentState(ok,rel);
  if(rel){lastReliableHr=heldHr;lastReliableAt=millis();}
  spo2Out=(ok&&isfinite(spo2Smooth))?clampf(spo2Smooth+spo2Offset,70.0f,100.0f):NAN;
  publishTelemetry(st, ok?heldHr:NAN);
  lastPublishSample=sampleCounter;
}

// ---------- Fast contact check (Model 1) on the most recent 100 samples ----------
static void updateContact() {
  static float cr[100], ci[100];
  for(int i=0;i<100;i++){
    uint32_t idx=(ringWrite+WINDOW_N-100+i)%WINDOW_N;
    cr[i]=rawRing[idx].red*CONTACT_COUNT_SCALE; ci[i]=rawRing[idx].ir*CONTACT_COUNT_SCALE;
  }
  const uint8_t prevKlass=lastContact.klass;
  ContactPrediction c=vitalSyncContactPredict(cr,ci);
  float recentMax=0; for(int i=75;i<100;i++) if(ci[i]>recentMax) recentMax=ci[i];
  if(recentMax<=CONTACT_FAST_IR_MAX) {          // finger just removed: react within 0.25 s
    c.klass=1; c.name="NO_CONTACT"; lastContact=c;
    contactPendingKlass=255; contactPendingCount=0;
  } else if(prevKlass==255 || c.klass==prevKlass) {
    lastContact=c; contactPendingKlass=255; contactPendingCount=0;
  } else {                                      // debounce other changes (2 checks = 0.5 s)
    if(c.klass==contactPendingKlass) contactPendingCount++; else {contactPendingKlass=c.klass;contactPendingCount=1;}
    if(contactPendingCount>=CONTACT_CONFIRM){lastContact=c;contactPendingKlass=255;contactPendingCount=0;}
    else {lastContact.irMax=c.irMax;lastContact.redMin=c.redMin;lastContact.irMean=c.irMean;lastContact.redMean=c.redMean;}
  }
  if(lastContact.klass==0) contactGoodRun+=CONTACT_HOP_N;
  if(lastContact.klass==1) {                    // finger off: discard everything learned about the old signal
    contactGoodRun=0; sqiWindowValid=false; sqiClass=2; sqiPoorRun=0; sqiGoodRun=0;
    sqiProb[0]=0; sqiProb[1]=0; sqiProb[2]=1; sqiReason="NO_CONTACT"; heldHr=NAN;
    m2Valid=false; spo2Smooth=NAN; spo2Out=NAN; spo2PI=NAN; spo2HistN=0; spo2HistPos=0;
    hrTrack=NAN; hrCand=NAN; hrCandCount=0; hrMissCount=0;
  }
  if(lastContact.klass!=prevKlass) publishNow();   // push contact changes to the dashboard immediately
}

// ---------- Precision HR: sub-sample peaks + fine spectrum + outlier-rejecting tracker ----------
static float hannW[WINDOW_N]; static bool hannInit=false;
static float xwA[WINDOW_N], xwB[WINDOW_N];
static const int HR_NB=261; static const float HR_F0=0.70f, HR_DF=0.01f;   // 42..198 bpm, 0.01 Hz step
static float gpA[HR_NB], gpB[HR_NB];

static void ensureHann(){ if(hannInit)return; for(int i=0;i<WINDOW_N;i++)hannW[i]=0.5f-0.5f*cosf(2.0f*PI_F*i/(WINDOW_N-1)); hannInit=true; }

static float goertzelPow(const float *x,float f) {
  float c=2.0f*cosf(2.0f*PI_F*f/FS_HZ), s1=0, s2=0;
  for(int n=0;n<WINDOW_N;n++){float s0=x[n]+c*s1-s2; s2=s1; s1=s0;}
  return s1*s1+s2*s2-c*s1*s2;
}
static float goertzelBand(const float *x,float flo,float fhi,float step) {
  float tot=0; if(flo<0.3f)flo=0.3f;
  for(float f=flo; f<=fhi+1e-6f; f+=step) tot+=goertzelPow(x,f);
  return tot;
}

// Peak-based HR with parabolic (sub-sample) peak timing, outlier-trimmed mean IBI. Picks the polarity with lowest IBI CV.
static float refinedPeakHr(const float *x) {
  int p[150]; float pr[150]; float best=NAN, bestCv=1e9f;
  for(int pol=0;pol<2;pol++) {
    float sg=pol==0?1.0f:-1.0f;
    int n=detectPeaks(x,WINDOW_N,sg,p,pr); if(n<4) continue;
    float t[150]; if(n>150)n=150;
    for(int i=0;i<n;i++) {
      int k=p[i]; float y0=sg*x[k-1],y1=sg*x[k],y2=sg*x[k+1];
      float den=y0-2*y1+y2, d=(fabsf(den)>1e-12f)?0.5f*(y0-y2)/den:0.0f;
      t[i]=k+clampf(d,-0.5f,0.5f);
    }
    float ibi[149]; int m=0;
    for(int i=0;i+1<n;i++){float d=(t[i+1]-t[i])/FS_HZ; if(d>=0.30f&&d<=1.50f)ibi[m++]=d;}
    if(m<3) continue;
    float tmp[149]; for(int i=0;i<m;i++)tmp[i]=ibi[i];
    float med=medianOf(tmp,m);
    float kept[149]; int kc=0; for(int i=0;i<m;i++) if(fabsf(ibi[i]-med)<=0.25f*med) kept[kc++]=ibi[i];
    if(kc<3) continue;
    float mu,sd; meanStd(kept,kc,mu,sd); float cv=sd/mu;
    if(cv<bestCv){bestCv=cv; best=60.0f/mu;}
  }
  return best;
}

// Fine-resolution spectral HR (Hann + Goertzel over 0.7-3.3 Hz, parabolic interpolation) using both channels.
static float spectralHr(const float *r,const float *ir,float &conf) {
  ensureHann(); conf=0;
  for(int i=0;i<WINDOW_N;i++){xwA[i]=r[i]*hannW[i]; xwB[i]=ir[i]*hannW[i];}
  double sa=0,sb=0;
  for(int k=0;k<HR_NB;k++){float f=HR_F0+HR_DF*k; gpA[k]=goertzelPow(xwA,f); gpB[k]=goertzelPow(xwB,f); sa+=gpA[k]; sb+=gpB[k];}
  if(sa<=1e-20||sb<=1e-20) return NAN;
  float tot=0,best=-1; int bk=0; static float comb[HR_NB];
  for(int k=0;k<HR_NB;k++){comb[k]=(float)(gpA[k]/sa+gpB[k]/sb); tot+=comb[k]; if(comb[k]>best){best=comb[k];bk=k;}}
  float delta=0;
  if(bk>0&&bk<HR_NB-1){float y0=comb[bk-1],y1=comb[bk],y2=comb[bk+1],den=y0-2*y1+y2; if(fabsf(den)>1e-12f)delta=clampf(0.5f*(y0-y2)/den,-0.5f,0.5f);}
  float nearPow=0; for(int k=bk-8;k<=bk+8;k++) if(k>=0&&k<HR_NB) nearPow+=comb[k];
  conf=nearPow/tot;
  return 60.0f*(HR_F0+HR_DF*(bk+delta));
}

static float computeHr(const float *f) {
  (void)f;
  float hrR=refinedPeakHr(wRed), hrI=refinedPeakHr(wIr);
  float specConf=0, hrSp=spectralHr(wRed,wIr,specConf);
  float hrPk=NAN;
  if(isfinite(hrR)&&isfinite(hrI)) {
    if(fabsf(hrR-hrI)<=6.0f) hrPk=0.5f*(hrR+hrI);
    else if(isfinite(hrSp)) hrPk=(fabsf(hrR-hrSp)<=fabsf(hrI-hrSp))?hrR:hrI;
    else hrPk=hrR;
  } else hrPk=isfinite(hrR)?hrR:hrI;
  float fused=NAN;
  if(isfinite(hrPk)&&isfinite(hrSp)) fused=(fabsf(hrPk-hrSp)<=8.0f)?(0.7f*hrPk+0.3f*hrSp):((specConf>0.35f)?hrSp:hrPk);
  else fused=isfinite(hrPk)?hrPk:hrSp;
  // Tracker: smooth small changes, hold through single outlier windows, accept a real jump after 2 consistent windows.
  if(!isfinite(fused)) { if(++hrMissCount>=2) hrTrack=NAN; }
  else {
    hrMissCount=0;
    if(!isfinite(hrTrack)) {hrTrack=fused; hrCandCount=0;}
    else if(fabsf(fused-hrTrack)<=12.0f) {hrTrack+=0.5f*(fused-hrTrack); hrCandCount=0;}
    else {
      if(hrCandCount>0&&fabsf(fused-hrCand)<=8.0f) hrCandCount++; else {hrCand=fused;hrCandCount=1;}
      if(hrCandCount>=2){hrTrack=fused;hrCandCount=0;}
    }
  }
  Serial.printf("# HR_DBG red=%.1f ir=%.1f spec=%.1f(conf %.2f) fused=%.1f track=%.1f\n",hrR,hrI,hrSp,specConf,fused,hrTrack);
  return hrTrack;
}

// SpO2: ratio-of-ratios with AC measured only at the heart-rate fundamental and 2nd harmonic (rejects broadband noise).
static void computeSpo2() {
  if(!isfinite(hrTrack)) return;
  ensureHann();
  for(int i=0;i<WINDOW_N;i++){xwA[i]=wRedHp[i]*hannW[i]; xwB[i]=wIrHp[i]*hannW[i];}
  float f0=hrTrack/60.0f;
  float pr=goertzelBand(xwA,f0-0.14f,f0+0.14f,0.02f)+goertzelBand(xwA,2*f0-0.20f,2*f0+0.20f,0.02f);
  float pi_=goertzelBand(xwB,f0-0.14f,f0+0.14f,0.02f)+goertzelBand(xwB,2*f0-0.20f,2*f0+0.20f,0.02f);
  double rd=0,id=0,iq=0;
  for(int i=0;i<WINDOW_N;i++){rd+=wRedRaw[i];id+=wIrRaw[i];iq+=(double)wIrHp[i]*wIrHp[i];}
  float rDc=(float)(rd/WINDOW_N), iDc=(float)(id/WINDOW_N);
  if(rDc<1000.0f||iDc<1000.0f||pr<=1e-12f||pi_<=1e-12f) return;
  spo2PI=100.0f*2.828f*sqrtf((float)(iq/WINDOW_N))/iDc;
  if(spo2PI<0.05f) return;                                   // too little pulsatile signal
  float R=(sqrtf(pr)/rDc)/(sqrtf(pi_)/iDc);
  if(!(R>=0.2f&&R<=2.0f)) return;
  float sp=clampf(SPO2_A*R*R+SPO2_B*R+SPO2_C,60.0f,105.0f);
  spo2Hist[spo2HistPos]=sp; spo2HistPos=(spo2HistPos+1)%5; if(spo2HistN<5)spo2HistN++;
  float tmp[5]; for(int i=0;i<spo2HistN;i++)tmp[i]=spo2Hist[i];
  float med=medianOf(tmp,spo2HistN);                          // median of last 5 windows rejects spikes
  spo2Smooth=isfinite(spo2Smooth)?(SPO2_ALPHA*med+(1.0f-SPO2_ALPHA)*spo2Smooth):med;
  Serial.printf("# SPO2_DBG R=%.3f raw=%.1f smooth=%.1f offset=%.1f PI=%.2f%%\n",R,sp,spo2Smooth,spo2Offset,spo2PI);
}

// ---------- Serial commands: one-point SpO2 calibration ----------
static String cmdBuf;
static void handleSerialCommands() {
  while(Serial.available()) {
    char c=(char)Serial.read();
    if(c=='\n'||c=='\r') {
      cmdBuf.trim();
      if(cmdBuf.startsWith("CAL_SPO2,")) {
        float ref=cmdBuf.substring(9).toFloat();
        if(ref>=70.0f&&ref<=100.0f&&isfinite(spo2Smooth)) {
          spo2Offset=clampf(ref-spo2Smooth,-15.0f,15.0f);
          prefs.putFloat("spo2_off",spo2Offset);
          Serial.printf("# SPO2_CAL_OK reference=%.1f offset=%.2f (saved)\n",ref,spo2Offset);
        } else Serial.println("# SPO2_CAL_FAILED need finger on, a valid SpO2 reading, and reference 70-100");
      } else if(cmdBuf=="CAL_SPO2_RESET") {
        spo2Offset=0; prefs.putFloat("spo2_off",0); Serial.println("# SPO2_CAL_RESET offset=0");
      }
      cmdBuf="";
    } else if(cmdBuf.length()<40) cmdBuf+=c;
  }
}

// ---------- Robust SQI: smoothing + plausibility gates + hysteresis ----------
static void updateSqi(const float *f) {
  if(contactGoodRun<SQI_SETTLE_SAMPLES){sqiWindowValid=false; sqiReason="SETTLING"; return;}
  const bool first=!sqiWindowValid;
  for(int i=0;i<3;i++) sqiProb[i]=first?lastM3.probabilities[i]:(SQI_ALPHA*lastM3.probabilities[i]+(1.0f-SQI_ALPHA)*sqiProb[i]);
  uint8_t cls=0; for(uint8_t i=1;i<3;i++) if(sqiProb[i]>sqiProb[cls]) cls=i;
  heldHr=computeHr(f);
  computeSpo2();
  bool hard=false; const char *why="OK";
  if(!isfinite(heldHr))                                  {hard=true;why="NO_HR";}
  else if(heldHr<HR_MIN_BPM||heldHr>HR_MAX_BPM)          {hard=true;why="HR_OUT_OF_RANGE";}
  else if(f[29]>CLIP_MAX_FRAC||f[30]>CLIP_MAX_FRAC)      {hard=true;why="SIGNAL_CLIPPING";}
  if(!hard && cls==VITALSYNC_M3_GOOD) {                  // soft downgrades GOOD -> ACCEPTABLE
    if(isfinite(f[2])&&f[2]>HR_DISAGREE_SOFT)            {cls=VITALSYNC_M3_ACCEPTABLE;why="RED_IR_MISMATCH";}
    else if(lastM2.predictedClass==2)                    {cls=VITALSYNC_M3_ACCEPTABLE;why="MOTION";}
  }
  if(hard) {
    sqiClass=VITALSYNC_M3_POOR; sqiPoorRun=SQI_CONFIRM_RUNS; sqiGoodRun=0;
  } else if(first) {
    sqiClass=cls; sqiPoorRun=0; sqiGoodRun=0;
    if(cls==VITALSYNC_M3_POOR) why="MODEL_POOR";
  } else if(cls==VITALSYNC_M3_POOR) {
    sqiPoorRun++; sqiGoodRun=0;
    if(sqiPoorRun>=SQI_CONFIRM_RUNS) sqiClass=VITALSYNC_M3_POOR;
    why="MODEL_POOR";
  } else {
    sqiGoodRun++; sqiPoorRun=0;
    if(sqiClass!=VITALSYNC_M3_POOR) sqiClass=cls;
    else if(sqiGoodRun>=SQI_CONFIRM_RUNS) sqiClass=cls;
    else why="RECOVERING";
  }
  sqiReason=why; sqiWindowValid=true;
}

static void predictWindow() {
  float features[VITALSYNC_M3_N_FEATURES];
  bool ok=extractFeatures(features);
  if(!ok) Serial.println("STATUS,M3_FEATURE_HAS_NAN_USING_MODEL_IMPUTER");
  if(STREAM_FEATURES) {
    Serial.print("M3_FEATURES");
    for(int i=0;i<VITALSYNC_M3_N_FEATURES;i++)Serial.printf(",%.6f",features[i]);
    Serial.println();
  }
  VitalSyncModel3Prediction result=vitalSyncModel3Predict(features); lastM3=result;
  float m2f[VITALSYNC_RF_N_FEATURES]; extractModel2Features(m2f);
  lastM2=vitalSyncModel2Predict(m2f);
  m2Valid=(lastContact.klass!=1)&&(contactGoodRun>=SQI_SETTLE_SAMPLES);
  Serial.printf("M2_PRED,%s,CLEAN=%.3f,MILD=%.3f,SEVERE=%.3f\n",vitalSyncModel2ClassName(lastM2.predictedClass),
    lastM2.probabilities[0],lastM2.probabilities[1],lastM2.probabilities[2]);
  updateSqi(features);
  Serial.printf("# CONTACT_DBG ir_max=%.0f (need >98525) red_min=%.0f class=%s good_run=%lu\n",
    lastContact.irMax,lastContact.redMin,lastContact.name,(unsigned long)contactGoodRun);
  Serial.printf("M3_PRED,%s,GOOD=%.5f,ACCEPTABLE=%.5f,POOR=%.5f,RAW=%s,STABLE=%s,REASON=%s\n",
    vitalSyncModel3ClassName(result.predictedClass),
    result.probabilities[0],result.probabilities[1],result.probabilities[2],
    vitalSyncModel3ClassName(result.predictedClass),vitalSyncModel3ClassName(sqiClass),sqiReason);
  publishNow();
}

static void acceptRawSample(const RawSample &s) {
  if(!calibrated) {
    calBuf[calCount++]=s;
    if(calCount>=CAL_N)finishCalibration();
    return;
  }
  processOneSample(s,true);
  sampleCounter++;
  if(validSamples>=100 && ++sinceContactCheck>=CONTACT_HOP_N) {sinceContactCheck=0;updateContact();}
  if(validSamples>=WINDOW_N) {
    sincePrediction++;
    if(sincePrediction>=HOP_N) {sincePrediction=0;predictWindow();}
  }
  if(sampleCounter-lastPublishSample>=HEARTBEAT_N) publishNow();   // keep the dashboard fresh
  if(STREAM_RAW)Serial.printf("M3_RAW,%lu,%.0f,%.0f,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f\n",
    (unsigned long)sampleCounter,s.red,s.ir,s.ax,s.ay,s.az,s.gx,s.gy,s.gz);
}

void setup() {
  Serial.begin(921600); delay(800);
  Serial.println("# VitalSync Model3 live-test firmware - experimental");
  Serial.println("# Wiring retained: MAX30102+MPU6050 I2C 21/22; DS18B20 GPIO4");
  Serial.println("# Keep device attached and reasonably still during 20-second reference calibration.");
  Wire.begin(PIN_SDA,PIN_SCL); Wire.setClock(400000);
  if(!mpuInit()){Serial.println("FATAL,MPU6050_NOT_FOUND");while(true)delay(1000);}
  if(!calibrateGyro()){Serial.println("FATAL,MPU6050_READ_FAILED");while(true)delay(1000);}
  if(!ppg.begin(Wire,I2C_SPEED_FAST)){Serial.println("FATAL,MAX30102_NOT_FOUND");while(true)delay(1000);}
  // SparkFun MAX3010x library: LED current, sample average, red+IR mode,
  // FIFO sample rate, pulse width, ADC range. 400/4 ~= 100 output samples/s.
  ppg.setup(LED_CURRENT,4,2,400,411,16384);   // 400/4 = 100 output samples/s
  ppg.setPulseAmplitudeRed(LED_CURRENT);
  ppg.setPulseAmplitudeIR(LED_CURRENT);
  ppg.setPulseAmplitudeGreen(0);
  calBuf=(RawSample*)calloc(CAL_N,sizeof(RawSample));
  if(!calBuf){Serial.println("FATAL,CALIBRATION_BUFFER_ALLOCATION_FAILED");while(true)delay(1000);}
  prefs.begin("vitalsync",false); spo2Offset=clampf(prefs.getFloat("spo2_off",0.0f),-15.0f,15.0f);
  Serial.printf("# SpO2 calibration offset=%.2f (send CAL_SPO2,<reference %%> to calibrate)\n",spo2Offset);
  ds.begin(); ds.setResolution(12); ds.setWaitForConversion(false);
  Serial.println("READY,STARTING_20S_CALIBRATION");
}

void loop() {
  ppg.check();
  while(ppg.available()) {
    RawSample s;
    s.red=(float)ppg.getFIFORed();
    s.ir=(float)ppg.getFIFOIR();
    ppg.nextSample();
    if(!mpuRead(s)) {
      Serial.println("WARN,MPU_READ_FAILED_SAMPLE_SKIPPED");
      continue;
    }
    acceptRawSample(s);
  }
  serviceTemp();
  handleSerialCommands();
}