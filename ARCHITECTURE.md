# VitalSync - System Architecture

VitalSync is a multi-bed monitoring prototype that tells a nurse whether each reading is trustworthy: whether contact is good, whether motion is corrupting the signal, and whether the processed PPG is usable. It is an engineering prototype for decision support, not a clinically validated diagnostic or alarm system.

Inference runs **on the ESP32**. The network carries results (status, reason codes, readings, timestamps), and the backend and dashboard store and display them.

## 1. System block diagram

```mermaid
flowchart TB
    subgraph NODE["Bed node (one per bed: BED_01, BED_02, BED_03)"]
        direction TB
        subgraph SENS["Sensors"]
            PPG["MAX30102<br/>red / IR PPG<br/>100 Hz, I2C 0x57"]
            IMU["MPU6050<br/>accel + gyro<br/>100 Hz, I2C 0x68"]
            TMP["DS18B20<br/>temperature<br/>1 Hz, 1-Wire"]
        end
        subgraph MCU["ESP32 DevKit V1"]
            ACQ["Acquisition<br/>monotonic timestamps<br/>sample counters, ring buffer<br/>integrity flags"]
            PROC["On-chip processing<br/>34 features, NLMS + MAF<br/>Models 1, 2, 3<br/>validity gate, HR / SpO2"]
            ACQ --> PROC
        end
        PPG -- "I2C" --> ACQ
        IMU -- "I2C" --> ACQ
        TMP -- "GPIO 4" --> ACQ
    end

    NET["Wi-Fi transport<br/>MQTT or HTTP<br/>results only"]
    API["Backend<br/>API / broker<br/>PostgreSQL"]
    UI["Nurse dashboard<br/>ward overview<br/>bed detail, event history"]

    PROC --> NET
    NET --> API
    API --> UI
```

## 2. Per-window processing flowchart (runs on the ESP32)

An 8 s window (800 samples at 100 Hz) is produced every 2 s.

```mermaid
flowchart TD
    A["Acquire synchronized red/IR PPG + IMU<br/>100 Hz"] --> B["Form 8 s window<br/>2 s hop, 800 samples"]
    B --> C{"Integrity OK?<br/>no gaps, no FIFO overflow"}
    C -- "no" --> X0["READING_WITHHELD<br/>data / device issue"]
    C -- "yes" --> M1["Model 1<br/>contact detection"]
    M1 --> D{"Optical contact?"}
    D -- "absent" --> X1["SENSOR_NOT_CONTACTED<br/>skip PPG path<br/>show reposition prompt"]
    D -- "weak" --> X2["CONTACT_UNSTABLE<br/>withhold by default"]
    D -- "good" --> M2["Model 2<br/>motion-artifact class<br/>clean / mild / severe"]
    M2 --> F1["Filter 1<br/>IMU-referenced NLMS"]
    F1 --> F2["Filter 2<br/>moving average"]
    F2 --> M3["Model 3<br/>post-filter signal quality"]
    M3 --> G{"Validity gate<br/>(deterministic rules)"}
    G -- "pass" --> E["Estimate HR + SpO2"]
    E --> V1["READING_VALID<br/>or READING_VALID_MOTION_RECOVERED"]
    G -- "fail, motion present" --> W1["READING_WITHHELD_MOTION"]
    G -- "fail, no motion" --> W2["READING_WITHHELD_SIGNAL_QUALITY"]

    T["DS18B20 read<br/>separate validity path"] --> TV{"Probe valid?"}
    TV -- "yes" --> TOK["Temperature shown"]
    TV -- "no" --> TBAD["TEMPERATURE_UNAVAILABLE"]

    V1 --> P["Publish status, reason codes,<br/>readings, timestamps"]
    W1 --> P
    W2 --> P
    X0 --> P
    X1 --> P
    X2 --> P
    TOK --> P
    TBAD --> P
    P --> UI["Nurse dashboard"]
```

Model 1 and Model 3 are not duplicates. Model 1 asks whether the finger is optically coupled to the sensor. Model 3 asks whether the processed waveform is usable. The filters improve a waveform but do not prove it is accurate, so Model 3 and the validity gate judge the result after filtering.

## 3. Offline ML pipeline (how the models get onto the chip)

```mermaid
flowchart LR
    subgraph DATA["Data"]
        PTT["PhysioNet Pulse Transit Time PPG<br/>22 subjects, 66 records"]
        OWN["Own VitalSync recordings<br/>bed CSV logs"]
    end

    PTT --> RS["Resample 500 to 100 Hz<br/>8 s windows, 2 s hop"]
    OWN --> RS
    RS --> LAB["Labels<br/>ECG R-peak pulse timing<br/>(offline only, never a model input)"]
    RS --> FEAT["34-feature extraction<br/>fixed order"]
    LAB --> TRAIN
    FEAT --> TRAIN["Train Random Forest / XGBoost<br/>participant-grouped split + LOSO"]
    TRAIN --> EVAL["Evaluate<br/>macro-F1, severe recall<br/>false-accept rate"]
    EVAL --> EXP["Export to C<br/>small tree models"]
    EXP --> FW["ESP32 firmware<br/>same preprocessing + feature order"]
```

Rules that keep training and deployment consistent:
- ECG is used only to build labels offline. It is not available at deployment and is never a model input.
- The same preprocessing, feature calculations and feature order are applied in training and in firmware.
- Model 2 uses a zero-patient-calibration design, so a new patient needs no personal baseline recording.

## 4. Reading states shown on the dashboard

| Condition | State | Dashboard action |
|---|---|---|
| No fresh packet from node | `DEVICE_OFFLINE` | Show last communication time; not a patient condition |
| No contact | `SENSOR_NOT_CONTACTED` | Suppress HR/SpO2, show reposition instruction |
| Weak contact | `CONTACT_UNSTABLE` | Withhold until quality passes |
| Contact good, no artifact, SQI good | `READING_VALID` | Show fresh HR/SpO2 with timestamp |
| Motion artifact, filtering recovers, SQI passes | `READING_VALID_MOTION_RECOVERED` | Show values with motion/recovered badge |
| Motion artifact, SQI poor after filtering | `READING_WITHHELD_MOTION` | Withhold; ask for a steady hand |
| Contact good, no motion, SQI poor | `READING_WITHHELD_SIGNAL_QUALITY` | Withhold; explain the reason |
| Temperature probe invalid or stale | `TEMPERATURE_UNAVAILABLE` | Separate warning; PPG state unaffected |

A withheld reading never reuses an old value as current. The last reliable value is shown separately, with its exact timestamp.

## 5. Hardware summary

| Item | Choice |
|---|---|
| Controller | ESP32 DevKit V1 |
| Optical sensor | MAX30102, red/IR PPG, 100 Hz |
| Motion sensor | MPU6050, accel + gyro, 100 Hz, synchronized with PPG |
| Temperature | DS18B20, 1 Hz, probe/surface temperature (not core body temperature) |
| Window | 8 s, 2 s hop, 800 samples per channel |

## Safety boundary

VitalSync is a student engineering prototype. It is not intended to diagnose, treat, triage or independently alarm for medical emergencies, and it must not replace approved clinical monitoring or clinician judgment. SpO2 values are experimental estimates and are not clinically validated.
