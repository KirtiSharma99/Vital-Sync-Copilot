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
#include "VitalSync_Model3_Types.h"  // Types needed by Arduino auto-generated prototypes
#include "VitalSync_Model3_RF_ESP32.h"

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
static const uint8_t LED_CURRENT = 0x24;
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
  float rHp=hpRed.process(s.red);
  float iHp=hpIr.process(s.ir);
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

static void predictWindow() {
  float features[VITALSYNC_M3_N_FEATURES];
  bool ok=extractFeatures(features);
  if(!ok) Serial.println("STATUS,M3_FEATURE_HAS_NAN_USING_MODEL_IMPUTER");
  if(STREAM_FEATURES) {
    Serial.print("M3_FEATURES");
    for(int i=0;i<VITALSYNC_M3_N_FEATURES;i++)Serial.printf(",%.6f",features[i]);
    Serial.println();
  }
  VitalSyncModel3Prediction result=vitalSyncModel3Predict(features);
  Serial.printf("M3_PRED,%s,GOOD=%.5f,ACCEPTABLE=%.5f,POOR=%.5f\n",
    vitalSyncModel3ClassName(result.predictedClass),
    result.probabilities[0],result.probabilities[1],result.probabilities[2]);
}

static void acceptRawSample(const RawSample &s) {
  if(!calibrated) {
    calBuf[calCount++]=s;
    if(calCount>=CAL_N)finishCalibration();
    return;
  }
  processOneSample(s,true);
  sampleCounter++;
  if(validSamples>=WINDOW_N) {
    sincePrediction++;
    if(sincePrediction>=HOP_N) {sincePrediction=0;predictWindow();}
  }
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
  ppg.setup(LED_CURRENT,4,2,400,411,16384);
  calBuf=(RawSample*)calloc(CAL_N,sizeof(RawSample));
  if(!calBuf){Serial.println("FATAL,CALIBRATION_BUFFER_ALLOCATION_FAILED");while(true)delay(1000);}
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
}
