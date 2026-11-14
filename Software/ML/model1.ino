
#include <Wire.h>
#include "MAX30105.h"
#include "vitalsync_model1_model.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// ================= PIN CONFIGURATION =================

#define PIN_SDA 21
#define PIN_SCL 22

// ================= SENSOR =================

MAX30105 maxSensor;

// ================= ML MODEL =================

Eloquent::ML::Port::DecisionTree clf;

// ================= DATA =================

#define WINDOW_SIZE 100
#define PREDICTION_STEP 25

float redData[WINDOW_SIZE];
float irData[WINDOW_SIZE];

int sampleCount = 0;

// ================= MEDIAN =================

float findMedian(float data[], int size) {
  float temp[WINDOW_SIZE];

  memcpy(temp, data, sizeof(float) * size);

  qsort(temp, size, sizeof(float),
    [](const void *a, const void *b) {
      float x = *(float *)a;
      float y = *(float *)b;
      return (x > y) - (x < y);
    });

  if (size % 2 == 0) {
    return (temp[size / 2 - 1] + temp[size / 2]) / 2.0;
  }

  return temp[size / 2];
}

// ================= FEATURE EXTRACTION =================

void extractFeatures(float red[], float ir[], float features[]) {

  float redSum = 0, irSum = 0;

  float redMin = red[0], redMax = red[0];
  float irMin = ir[0], irMax = ir[0];

  for (int i = 0; i < WINDOW_SIZE; i++) {
    redSum += red[i];
    irSum += ir[i];

    if (red[i] < redMin) redMin = red[i];
    if (red[i] > redMax) redMax = red[i];

    if (ir[i] < irMin) irMin = ir[i];
    if (ir[i] > irMax) irMax = ir[i];
  }

  float redMean = redSum / WINDOW_SIZE;
  float irMean = irSum / WINDOW_SIZE;

  float redVariance = 0, irVariance = 0;
  float covariance = 0;

  float redDiffSum = 0, irDiffSum = 0;
  float redDiffSquareSum = 0, irDiffSquareSum = 0;

  for (int i = 0; i < WINDOW_SIZE; i++) {

    float rd = red[i] - redMean;
    float id = ir[i] - irMean;

    redVariance += rd * rd;
    irVariance += id * id;

    covariance += rd * id;

    if (i > 0) {
      float redDiff = red[i] - red[i - 1];
      float irDiff = ir[i] - ir[i - 1];

      redDiffSum += redDiff;
      irDiffSum += irDiff;

      redDiffSquareSum += redDiff * redDiff;
      irDiffSquareSum += irDiff * irDiff;
    }
  }

  float redStd = sqrt(redVariance / WINDOW_SIZE);
  float irStd = sqrt(irVariance / WINDOW_SIZE);

  float redDiffMean = redDiffSum / (WINDOW_SIZE - 1);
  float irDiffMean = irDiffSum / (WINDOW_SIZE - 1);

  float redDiffStd = sqrt(
    (redDiffSquareSum / (WINDOW_SIZE - 1)) -
    (redDiffMean * redDiffMean)
  );

  float irDiffStd = sqrt(
    (irDiffSquareSum / (WINDOW_SIZE - 1)) -
    (irDiffMean * irDiffMean)
  );

  float correlation = 0;

  if (redStd > 0 && irStd > 0) {
    correlation = covariance /
      (WINDOW_SIZE * redStd * irStd);
  }

  float redMedian = findMedian(red, WINDOW_SIZE);
  float irMedian = findMedian(ir, WINDOW_SIZE);

  float redACRMS = redStd;
  float irACRMS = irStd;

  float redIRRatio = redMean / (irMean + 0.000001);

  // Exact feature order used in Python training
  features[0]  = redMean;
  features[1]  = redStd;
  features[2]  = redMin;
  features[3]  = redMax;
  features[4]  = redMax - redMin;
  features[5]  = redMedian;

  features[6]  = irMean;
  features[7]  = irStd;
  features[8]  = irMin;
  features[9]  = irMax;
  features[10] = irMax - irMin;
  features[11] = irMedian;

  features[12] = redDiffStd;
  features[13] = irDiffStd;

  features[14] = redACRMS;
  features[15] = irACRMS;

  features[16] = redIRRatio;
  features[17] = correlation;
}

// ================= SETUP =================

void setup() {

  Serial.begin(115200);

  Wire.begin(PIN_SDA, PIN_SCL);

  Serial.println("VitalSync Model 1 Starting...");

  if (!maxSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not detected!");
    while (1);
  }

  maxSensor.setup(0x1F, 1, 2, 100, 411, 4096);

  Serial.println("MAX30102 initialized!");
  Serial.println("Place finger on sensor.");
}

// ================= LOOP =================

void loop() {

  maxSensor.check();

  while (maxSensor.available()) {

    uint32_t redValue = maxSensor.getRed();
    uint32_t irValue = maxSensor.getIR();

    redData[sampleCount] = (float)redValue;
    irData[sampleCount] = (float)irValue;

    sampleCount++;

    maxSensor.nextSample();

    if (sampleCount >= WINDOW_SIZE) {

  float features[18];

  extractFeatures(redData, irData, features);

  int prediction = clf.predict(features);

  Serial.print("Finger Placement: ");

  if (prediction == 0) {
    Serial.println("GOOD");
  }
  else if (prediction == 1) {
    Serial.println("NO_FINGER");
  }
  else if (prediction == 2) {
    Serial.println("POOR");
  }
  else {
    Serial.println("UNKNOWN");
  }

  // Slide the window by 25 samples.
  // Next prediction comes after 250 ms.
  for (int i = 0; i < WINDOW_SIZE - PREDICTION_STEP; i++) {
    redData[i] = redData[i + PREDICTION_STEP];
    irData[i] = irData[i + PREDICTION_STEP];
  }

  sampleCount = WINDOW_SIZE - PREDICTION_STEP;
}
  }
}
