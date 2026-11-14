#pragma once
// Shared result types used by Model3 feature-extraction functions.
// Kept in a header because Arduino IDE auto-generates function prototypes
// before the main .ino body; return types must be visible before those prototypes.

struct BeatResult {
  float hr, ibiCv, ibiRmssd, templateCorr, promMean, promCv;
  int peakCount;
  int peaks[40];
};

struct SpectralResult {
  float dom, peakRatio, entropy, bandPower;
};
