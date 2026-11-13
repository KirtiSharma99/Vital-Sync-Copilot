/*
  VitalSync BED_01 - ESP32 + MAX30102 + MPU6050 + DS18B20
  Model 2: 3-class PPG motion-artifact classifier (Clean / Mild / Severe)

  Live acquisition: nominal 100 Hz; inference window 800 samples (8 s); hop 200 (2 s).
  Random Forest is embedded in VitalSync_Model2_RF_ESP32.h (100-tree compact model).

  v2 FIRMWARE FIXES: zero-phase Butterworth bandpass, notch-proof peak detector with median-IBI heart rate,
  band-passed AC/DC ratio, gyro converted to rad/s for the model. Still PROVISIONAL (see note below).

  IMPORTANT VALIDATION NOTE:
  The uploaded training package contained the 34-feature schema and trained RF,
  but not the original Python feature-generation source. The routines below
  implement the meanings in that schema as an ON-DEVICE PROVISIONAL port.
  They are not claimed to be bit-for-bit identical to the training pipeline.
  Do not interpret provisional labels as validated Model 2 performance until
  these 34 features are cross-checked against the original pipeline/reference rows.

  Acceleration note: PhysioNet raw a_x/a_y/a_z examples are around +/-9.8 at rest,
  i.e. numerically m/s^2, although the exported feature names carry a _g suffix.
  We therefore convert the MPU6050 g readings to m/s^2 before calculating the
  model inputs to match the numerical scale seen in the training exports.

  Serial:
    PRED_PROVISIONAL,class,p_clean,p_mild,p_severe
    FEAT_PROVISIONAL,<34 comma-separated feature values in model order>
    S,... raw sample rows only when STREAM_RAW is set to 1
*/

#include <Arduino.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <math.h>
#include "MAX30105.h"
#include "esp_timer.h"
#include "VitalSync_Model2_RF_ESP32.h"

#define PIN_SDA 21
#define PIN_SCL 22
#define PIN_DS18B20 4
#define MPU_ADDR 0x68
#define FS_HZ 100.0f
#define TICK_US 10000LL
#define WINDOW_N 800
#define HOP_N 200
#define ACC_LSB_PER_G 8192.0f       // MPU6050 +/-4 g
#define GYR_LSB_PER_DPS 65.5f       // MPU6050 +/-500 deg/s
#define G_TO_MS2 9.80665f
#define GYRO_TO_MODEL 0.01745329f   // training gyro columns look like rad/s (set 1.0f if your dataset header says deg/s)
#define LED_CURRENT 0x24
#define STREAM_RAW 0                // set to 1 for S,... rows (very verbose)

MAX30105 ppg;
OneWire oneWire(PIN_DS18B20);
DallasTemperature ds(&oneWire);

struct Sample {
  uint32_t idx;
  int64_t t_us;
  float red, ir;
  float ax, ay, az;                 // m/s^2 to match PhysioNet numeric scale
  float gx, gy, gz;                 // degrees/s, bias-corrected
  float temp_c;
};

// Big buffers live on the heap (allocated in setup): static .bss on the classic ESP32 is limited to ~160 KB.
static Sample *ringBuf = nullptr;
static uint32_t sampleIdx = 0;
static uint32_t sincePred = 0;
static int64_t firstSampleUs = 0;
static int64_t nextTickUs = 0;
static float gyroBiasX = 0, gyroBiasY = 0, gyroBiasZ = 0;
static float tempC = NAN;
static bool tempPending = false;
static uint32_t lastTempReqMs = 0;
// Large work buffers are static: ESP32 loop-task stack is intentionally kept small.
static Sample *orderedBuf = nullptr;
static float *rawRed = nullptr, *rawIr = nullptr, *redBP = nullptr, *irBP = nullptr;
static float *accDyn = nullptr, *gyroMag = nullptr, *gyroDyn = nullptr, *accDemean = nullptr;
static float (*templateBuf)[64] = nullptr;
static float *padBuf = nullptr;   // filter work buffer (WINDOW_N + 2*PPG_PAD)

static bool allocBuffers() {
  ringBuf    = (Sample*)calloc(WINDOW_N, sizeof(Sample));
  orderedBuf = (Sample*)calloc(WINDOW_N, sizeof(Sample));
  rawRed   = (float*)calloc(WINDOW_N, sizeof(float));
  rawIr    = (float*)calloc(WINDOW_N, sizeof(float));
  redBP    = (float*)calloc(WINDOW_N, sizeof(float));
  irBP     = (float*)calloc(WINDOW_N, sizeof(float));
  accDyn   = (float*)calloc(WINDOW_N, sizeof(float));
  gyroMag  = (float*)calloc(WINDOW_N, sizeof(float));
  gyroDyn  = (float*)calloc(WINDOW_N, sizeof(float));
  accDemean= (float*)calloc(WINDOW_N, sizeof(float));
  templateBuf = (float(*)[64])calloc(20*64, sizeof(float));
  padBuf = (float*)calloc(WINDOW_N+2*150, sizeof(float));
  return ringBuf && orderedBuf && rawRed && rawIr && redBP && irBP &&
         accDyn && gyroMag && gyroDyn && accDemean && templateBuf && padBuf;
}

static inline float sqf(float x) { return x*x; }

bool mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR); Wire.write(reg); Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool mpuInit() {
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x75);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1) != 1) return false;
  uint8_t who = Wire.read();
  Serial.printf("# MPU6050 WHO_AM_I=0x%02X\n", who);
  bool ok = true;
  ok &= mpuWrite(0x6B, 0x01); delay(50); // wake, PLL clock
  ok &= mpuWrite(0x19, 9);                // 100 Hz sample rate
  ok &= mpuWrite(0x1A, 0x03);             // DLPF
  ok &= mpuWrite(0x1B, 0x08);             // +/-500 dps
  ok &= mpuWrite(0x1C, 0x08);             // +/-4 g
  return ok;
}

static inline int16_t readS16(const uint8_t *b, int i) {
  return (int16_t)(((uint16_t)b[i] << 8) | b[i+1]);
}

bool mpuRead(Sample &s) {
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14) != 14) return false;
  uint8_t b[14]; for (int i=0; i<14; ++i) b[i] = Wire.read();
  s.ax = (readS16(b,0) / ACC_LSB_PER_G) * G_TO_MS2;
  s.ay = (readS16(b,2) / ACC_LSB_PER_G) * G_TO_MS2;
  s.az = (readS16(b,4) / ACC_LSB_PER_G) * G_TO_MS2;
  s.gx = readS16(b,8) / GYR_LSB_PER_DPS - gyroBiasX;
  s.gy = readS16(b,10) / GYR_LSB_PER_DPS - gyroBiasY;
  s.gz = readS16(b,12) / GYR_LSB_PER_DPS - gyroBiasZ;
  return true;
}

bool calibrateGyro() {
  const int N = 500; double sx=0, sy=0, sz=0;
  Serial.println("# Keep the IMU completely still for gyro bias calibration (5 s)");
  for (int i=0; i<N; ++i) {
    Sample s;
    if (!mpuRead(s)) return false;
    sx += s.gx; sy += s.gy; sz += s.gz;
    delay(10);
  }
  gyroBiasX = sx/N; gyroBiasY = sy/N; gyroBiasZ = sz/N;
  Serial.printf("# gyro_bias_dps=%.4f,%.4f,%.4f\n", gyroBiasX, gyroBiasY, gyroBiasZ);
  return true;
}

void serviceTemp() {
  uint32_t now = millis();
  if (!tempPending && now-lastTempReqMs >= 1000) {
    ds.requestTemperatures(); lastTempReqMs=now; tempPending=true;
  } else if (tempPending && now-lastTempReqMs >= 800) {
    float t=ds.getTempCByIndex(0);
    tempC=(t==DEVICE_DISCONNECTED_C || t < -50 || t > 125) ? NAN : t;
    tempPending=false;
  }
}

// ---------- Basic statistics ----------
void meanStd(const float *x, int n, float &mean, float &sd) {
  double s=0; for(int i=0;i<n;i++) s+=x[i];
  mean=(float)(s/n);
  double v=0; for(int i=0;i<n;i++) v+=sqf(x[i]-mean);
  sd=sqrtf((float)(v/n));
}

float corrSigned(const float *a, const float *b, int n) {
  float ma,sa,mb,sb; meanStd(a,n,ma,sa); meanStd(b,n,mb,sb);
  if(sa < 1e-9f || sb < 1e-9f) return 0;
  double c=0; for(int i=0;i<n;i++) c+=(a[i]-ma)*(b[i]-mb);
  return (float)(c/(n*sa*sb));
}

void shapeStats(const float *x, int n, float &skew, float &kurt) {
  float m,sd; meanStd(x,n,m,sd);
  if(sd < 1e-9f) { skew=0; kurt=0; return; }
  double m3=0,m4=0;
  for(int i=0;i<n;i++) { double z=(x[i]-m)/sd; m3+=z*z*z; m4+=z*z*z*z; }
  skew=(float)(m3/n); kurt=(float)(m4/n-3.0);
}

// ---------- Direct DFT band spectrum features ----------
// Use exactly N=800 sample-length frequency bins (100/800 = 0.125 Hz),
// which matches the frequency increments visible in the saved feature matrix.
float spectralFeatures(const float *x, float loHz, float hiHz, float *peakRatioOut, float *entropyOut) {
  float m,sd; meanStd(x,WINDOW_N,m,sd);
  int k0=(int)ceilf(loHz*WINDOW_N/FS_HZ), k1=(int)floorf(hiHz*WINDOW_N/FS_HZ);
  if(k0<1)k0=1; if(k1>WINDOW_N/2-1)k1=WINDOW_N/2-1;
  int count=k1-k0+1;
  if(count<=1){if(peakRatioOut)*peakRatioOut=0;if(entropyOut)*entropyOut=0;return 0;}
  double powers[400]; double total=0,maxP=-1; int maxK=k0;
  for(int k=k0;k<=k1;k++) {
    const float angle=-2.0f*PI*k/WINDOW_N;
    const float cs=cosf(angle), sn=sinf(angle);
    float wr=1.0f, wi=0.0f; double re=0,im=0;
    for(int n=0;n<WINDOW_N;n++) {
      float hann=0.5f-0.5f*cosf(2.0f*PI*n/(WINDOW_N-1));
      float v=(x[n]-m)*hann;
      re += v*wr; im += v*wi;
      float nw=wr*cs-wi*sn; wi=wr*sn+wi*cs; wr=nw;
    }
    double power=re*re+im*im; powers[k-k0]=power; total+=power;
    if(power>maxP){maxP=power;maxK=k;}
  }
  if(total<=1e-20){if(peakRatioOut)*peakRatioOut=0;if(entropyOut)*entropyOut=0;return 0;}
  double entropy=0;
  for(int j=0;j<count;j++){double p=powers[j]/total;if(p>1e-15)entropy-=p*log(p);}
  if(peakRatioOut)*peakRatioOut=(float)(maxP/total);
  if(entropyOut)*entropyOut=(float)(entropy/log((double)count));
  return (float)maxK*FS_HZ/WINDOW_N;
}

// 0.5-5 Hz zero-phase Butterworth bandpass (scipy butter(2,[0.5,5],'band',fs=100) as SOS, applied
// forward+backward like filtfilt, with odd-extension padding so window edges do not ring).
#define SOS_SECTIONS 2
#define PPG_PAD 150
static const float SOS_COEF[SOS_SECTIONS][6] = {
  {0.0165819317f, 0.0331638633f, 0.0165819317f, 1.0f, -1.6288507686f, 0.6994634776f},
  {1.0f, -2.0f, 1.0f, 1.0f, -1.9573890395f, 0.9585316852f}
};
static void sosPass(float *x, int n, bool reverse) {
  for(int s=0;s<SOS_SECTIONS;s++) {
    const double b0=SOS_COEF[s][0],b1=SOS_COEF[s][1],b2=SOS_COEF[s][2],a1=SOS_COEF[s][4],a2=SOS_COEF[s][5];
    double z1=0,z2=0;
    for(int k=0;k<n;k++) {
      int i=reverse?(n-1-k):k;
      double xi=x[i], y=b0*xi+z1;
      z1=b1*xi-a1*y+z2; z2=b2*xi-a2*y; x[i]=(float)y;
    }
  }
}
void bandpassPPG(const float *in, float *out) {
  float *e=padBuf; const int L=WINDOW_N+2*PPG_PAD;
  double s=0; for(int i=0;i<WINDOW_N;i++) s+=in[i];
  const float m=(float)(s/WINDOW_N);
  for(int i=0;i<WINDOW_N;i++) e[PPG_PAD+i]=in[i]-m;
  for(int k=1;k<=PPG_PAD;k++) {
    e[PPG_PAD-k]=2.0f*e[PPG_PAD]-e[PPG_PAD+k];
    e[PPG_PAD+WINDOW_N-1+k]=2.0f*e[PPG_PAD+WINDOW_N-1]-e[PPG_PAD+WINDOW_N-1-k];
  }
  sosPass(e,L,false); sosPass(e,L,true);
  for(int i=0;i<WINDOW_N;i++) out[i]=e[PPG_PAD+i];
}

// Pulse peak detection on a zero-mean band-passed window.
// local maximum over +/-12 samples (0.24 s) -> removes dicrotic-notch bumps; height gate; 0.33 s minimum spacing.
int detectPulsePeaks(const float *x, int *peaks, float pol) {
  const int W=12, MIN_DIST=33, MAXC=100;
  int cand[MAXC]; float h[MAXC]; int nc=0;
  for(int i=1;i<WINDOW_N-1 && nc<MAXC;i++) {
    const float v=pol*x[i]; if(v<=0) continue;
    bool isMax=true; int a=i-W<0?0:i-W, b=i+W>WINDOW_N-1?WINDOW_N-1:i+W;
    for(int j=a;j<=b;j++) if(j!=i && pol*x[j]>v){isMax=false;break;}
    if(isMax){cand[nc]=i;h[nc]=v;nc++;}
  }
  if(nc==0) return 0;
  float tmp[MAXC]; for(int i=0;i<nc;i++) tmp[i]=h[i];
  for(int i=1;i<nc;i++){float k=tmp[i];int j=i-1;while(j>=0&&tmp[j]>k){tmp[j+1]=tmp[j];j--;}tmp[j+1]=k;}
  const float thr=0.30f*tmp[(nc*3)/4];
  int count=0, last=-1000;
  for(int c=0;c<nc;c++) {
    if(h[c]<thr) continue;
    if(count>0 && cand[c]-last<MIN_DIST) { if(h[c]>pol*x[peaks[count-1]]) {peaks[count-1]=cand[c]; last=cand[c];} }
    else if(count<40) { peaks[count++]=cand[c]; last=cand[c]; }
  }
  return count;
}

static void ibiStats(const int *p, int n, float &hr, float &cv) {
  hr=0; cv=1e9f; if(n<2) return;
  float ibi[39]; int m=n-1; if(m>39)m=39;
  for(int i=0;i<m;i++) ibi[i]=(p[i+1]-p[i])/FS_HZ;
  float mean,sd; meanStd(ibi,m,mean,sd); cv=(mean>0)?sd/mean:1e9f;
  float s[39]; for(int i=0;i<m;i++) s[i]=ibi[i];
  for(int i=1;i<m;i++){float k=s[i];int j=i-1;while(j>=0&&s[j]>k){s[j+1]=s[j];j--;}s[j+1]=k;}
  const float med=(m&1)?s[m/2]:0.5f*(s[m/2-1]+s[m/2]);
  hr=(med>0)?60.0f/med:0;
}

float peakFeatures(const float *x, int *peaks, int &nPeaks, float &intervalCv) {
  int p1[40],p2[40];
  int n1=detectPulsePeaks(x,p1,1.0f), n2=detectPulsePeaks(x,p2,-1.0f);
  float hr1,cv1,hr2,cv2; ibiStats(p1,n1,hr1,cv1); ibiStats(p2,n2,hr2,cv2);
  // Choose the polarity whose beat intervals are more regular (the notch side is less regular).
  bool use1;
  if(n1<3 && n2>=3) use1=false; else if(n2<3 && n1>=3) use1=true; else use1=(cv1<=cv2);
  nPeaks=use1?n1:n2; for(int i=0;i<nPeaks;i++) peaks[i]=use1?p1[i]:p2[i];
  if(nPeaks<2){intervalCv=0;return 0;}
  intervalCv=use1?cv1:cv2; return use1?hr1:hr2;
}

float beatTemplateCorrelation(const float *x, const int *peaks, int nPeaks) {
  const int L=64, MAX_T=20;
  int nt=0;
  for(int b=0;b<nPeaks-1 && nt<MAX_T;b++) {
    int a=peaks[b], z=peaks[b+1]; if(z-a<30 || z-a>200) continue;
    for(int j=0;j<L;j++) {
      float pos=a+(z-a)*(float)j/(L-1); int k=(int)pos; float f=pos-k;
      templateBuf[nt][j]=x[k]*(1-f)+x[(k+1<WINDOW_N?k+1:k)]*f;
    }
    float m,sd; meanStd(templateBuf[nt],L,m,sd); if(sd>1e-8f) nt++;
  }
  if(nt<2) return 0;
  float vals[190]; int nv=0;
  for(int a=0;a<nt;a++) for(int b=a+1;b<nt;b++) vals[nv++]=corrSigned(templateBuf[a],templateBuf[b],L);
  // Median of pairwise correlations.
  for(int i=0;i<nv;i++) for(int j=i+1;j<nv;j++) if(vals[j]<vals[i]) {float q=vals[i];vals[i]=vals[j];vals[j]=q;}
  return vals[nv/2];
}

// Returns true when 34 finite schema-ordered inputs could be constructed.
bool extractModel2Features(const Sample *w, float f[34]) {
  double axm=0,aym=0,azm=0;
  for(int i=0;i<WINDOW_N;i++){ axm+=w[i].ax; aym+=w[i].ay; azm+=w[i].az; }
  axm/=WINDOW_N; aym/=WINDOW_N; azm/=WINDOW_N;
  double accSq=0; float accPeak=0;
  for(int i=0;i<WINDOW_N;i++) {
    rawRed[i]=w[i].red; rawIr[i]=w[i].ir;
    float dx=w[i].ax-(float)axm, dy=w[i].ay-(float)aym, dz=w[i].az-(float)azm;
    accDyn[i]=sqrtf(dx*dx+dy*dy+dz*dz);
    accSq+=sqf(accDyn[i]); if(accDyn[i]>accPeak)accPeak=accDyn[i];
    gyroMag[i]=GYRO_TO_MODEL*sqrtf(w[i].gx*w[i].gx+w[i].gy*w[i].gy+w[i].gz*w[i].gz);
  }
  f[0]=sqrtf((float)(accSq/WINDOW_N)); f[1]=accPeak;
  float dummy1,dummy2;
  f[2]=spectralFeatures(accDyn,0.3f,10.0f,&dummy1,&dummy2);
  float gm,gs; meanStd(gyroMag,WINDOW_N,gm,gs);
  double g2=0; for(int i=0;i<WINDOW_N;i++)g2+=sqf(gyroMag[i]);
  f[3]=sqrtf((float)(g2/WINDOW_N)); f[4]=gs;
  for(int i=0;i<WINDOW_N;i++)gyroDyn[i]=gyroMag[i]-gm;
  f[5]=spectralFeatures(gyroDyn,0.3f,10.0f,&dummy1,&dummy2);

  bandpassPPG(rawRed,redBP); bandpassPPG(rawIr,irBP);
  int rPeaks[40],iPeaks[40], nr=0,ni=0; float rCv=0,iCv=0;
  f[6]=peakFeatures(redBP,rPeaks,nr,rCv);
  f[7]=peakFeatures(irBP,iPeaks,ni,iCv);
  f[8]=fabsf(f[6]-f[7]);
  float rMean,rSd,iMean,iSd; meanStd(rawRed,WINDOW_N,rMean,rSd); meanStd(rawIr,WINDOW_N,iMean,iSd);
  float rbm,rbs,ibm,ibs; meanStd(redBP,WINDOW_N,rbm,rbs); meanStd(irBP,WINDOW_N,ibm,ibs);
  float rRatio=(fabsf(rMean)>1e-6f)?rbs/fabsf(rMean):0;
  float iRatio=(fabsf(iMean)>1e-6f)?ibs/fabsf(iMean):0;
  f[9]=0.5f*(rRatio+iRatio);
  float rPr,rEn,iPr,iEn;
  f[23]=spectralFeatures(redBP,0.5f,5.0f,&rPr,&rEn);
  f[33]=spectralFeatures(irBP,0.5f,5.0f,&iPr,&iEn);
  f[10]=fabsf(f[23]-f[2]); f[11]=fabsf(f[33]-f[2]);
  f[12]=fabsf(f[23]-f[5]); f[13]=fabsf(f[33]-f[5]);
  f[14]=rCv; f[15]=(float)nr; f[16]=rPr; f[17]=rEn;
  f[18]=beatTemplateCorrelation(redBP,rPeaks,nr);
  shapeStats(redBP,WINDOW_N,f[19],f[20]);
  // Signed Pearson correlations are retained because negative values exist in the saved training matrix.
  float accMean,accSd; meanStd(accDyn,WINDOW_N,accMean,accSd);
  for(int i=0;i<WINDOW_N;i++) accDemean[i]=accDyn[i]-accMean;
  f[21]=corrSigned(redBP,accDemean,WINDOW_N);
  f[22]=corrSigned(redBP,gyroDyn,WINDOW_N);
  f[24]=iCv; f[25]=(float)ni; f[26]=iPr; f[27]=iEn;
  f[28]=beatTemplateCorrelation(irBP,iPeaks,ni);
  shapeStats(irBP,WINDOW_N,f[29],f[30]);
  f[31]=corrSigned(irBP,accDemean,WINDOW_N);
  f[32]=corrSigned(irBP,gyroDyn,WINDOW_N);
  for(int i=0;i<34;i++) if(!isfinite(f[i])) return false;
  if(nr<3 || ni<3 || f[6]<35 || f[6]>210 || f[7]<35 || f[7]>210) return false;
  return true;
}
void predictWindow() {
  uint32_t oldest=sampleIdx-WINDOW_N;
  for(uint32_t i=0;i<WINDOW_N;i++) orderedBuf[i]=ringBuf[(oldest+i)%WINDOW_N];
  float features[34];
  bool ok=extractModel2Features(orderedBuf,features);
  // Signal sanity line: DC should be ~50k-200k (never above 262143), AC (band-passed std) clearly > 0,
  // and the effective sample rate should be ~100 Hz. If not, fix the optics/timing before trusting the model.
  float rdm,rds,idm,ids,rbm,rbs,ibm,ibs;
  meanStd(rawRed,WINDOW_N,rdm,rds); meanStd(rawIr,WINDOW_N,idm,ids);
  meanStd(redBP,WINDOW_N,rbm,rbs); meanStd(irBP,WINDOW_N,ibm,ibs);
  const float fsEff=(float)sampleIdx*1.0e6f/(float)(esp_timer_get_time()-firstSampleUs);
  Serial.printf("# SIG red_dc=%.0f ir_dc=%.0f red_ac=%.1f ir_ac=%.1f fs_eff_hz=%.1f\n",rdm,idm,rbs,ibs,fsEff);
  if(!ok) {
    Serial.printf("STATUS,WINDOW_REJECTED,hr_red=%.1f,hr_ir=%.1f,peaks_red=%.0f,peaks_ir=%.0f\n",
      features[6],features[7],features[15],features[25]);
    return;
  }
  Serial.print("FEAT_PROVISIONAL");
  for(int i=0;i<34;i++) Serial.printf(",%.6f",features[i]);
  Serial.println();
  VitalSyncRFResult r=vitalSyncModel2Predict(features);
  Serial.printf("PRED_PROVISIONAL,%s,%.5f,%.5f,%.5f\n",
    vitalSyncModel2ClassName(r.predictedClass),r.probabilities[0],r.probabilities[1],r.probabilities[2]);
}

void setup() {
  Serial.begin(921600); delay(800);
  Serial.println("# VitalSync BED_01 Model2 live-test firmware");
  Serial.println("# Output is PROVISIONAL until feature parity is checked.");
  if(!allocBuffers()){Serial.println("FATAL,OUT_OF_MEMORY");while(true)delay(1000);}
  Serial.printf("# free heap after buffers: %u bytes\n",(unsigned)ESP.getFreeHeap());
  Wire.begin(PIN_SDA,PIN_SCL); Wire.setClock(400000);
  if(!mpuInit()){Serial.println("FATAL,MPU6050_NOT_FOUND");while(true)delay(1000);}
  if(!calibrateGyro()){Serial.println("FATAL,MPU6050_READ_FAILED");while(true)delay(1000);}
  if(!ppg.begin(Wire,I2C_SPEED_FAST)){Serial.println("FATAL,MAX30102_NOT_FOUND");while(true)delay(1000);}
  // LED brightness, sample average, red+IR mode, FIFO sample rate, pulse width, ADC range.
  // 400 samples/s averaged by 4 gives ~100 samples/s at the FIFO output.
  ppg.setup(LED_CURRENT,4,2,400,411,16384);
  ds.begin(); ds.setResolution(12); ds.setWaitForConversion(false);
  Serial.println("# Ready: keep still first; first prediction follows 8 s; then every 2 s.");
  nextTickUs=esp_timer_get_time()+TICK_US;
}

void loop() {
  int64_t now;
  while((now=esp_timer_get_time())<nextTickUs){ if(nextTickUs-now>2000) delay(1); }
  int64_t tick=now; nextTickUs+=TICK_US; if(nextTickUs<now)nextTickUs=now+TICK_US;
  Sample imu; bool imuOk=mpuRead(imu); if(!imuOk){imu.ax=imu.ay=imu.az=0;imu.gx=imu.gy=imu.gz=0;}
  serviceTemp(); ppg.check();
  while(ppg.available()) {
    if(sampleIdx==0) firstSampleUs=tick;
    Sample s; s.idx=sampleIdx; s.t_us=tick;
    s.red=(float)ppg.getFIFORed(); s.ir=(float)ppg.getFIFOIR(); ppg.nextSample();
    s.ax=imu.ax;s.ay=imu.ay;s.az=imu.az;s.gx=imu.gx;s.gy=imu.gy;s.gz=imu.gz;s.temp_c=tempC;
    ringBuf[sampleIdx%WINDOW_N]=s;
    if(STREAM_RAW) Serial.printf("S,%lu,%lld,%.0f,%.0f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.2f\n",
      (unsigned long)sampleIdx,(long long)tick,s.red,s.ir,s.ax,s.ay,s.az,s.gx,s.gy,s.gz,s.temp_c);
    sampleIdx++; sincePred++;
    if(sampleIdx>=WINDOW_N && sincePred>=HOP_N){sincePred=0;predictWindow();}
  }
}
