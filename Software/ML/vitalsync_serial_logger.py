
import serial
import csv
from datetime import datetime

PORT = "COM13"  # Use your actual ESP32 COM port
BAUD = 115200

TARGET_SAMPLES = 30000

filename = "vitalsync_model1_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"

ser = serial.Serial(PORT, BAUD, timeout=1)

count = 0

with open(filename, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "timestamp_ms",
        "sample_no",
        "red_raw",
        "ir_raw",
        "label"
    ])

    print("Recording started...")
    print("Saving file:", filename)
    print("Target:", TARGET_SAMPLES)

    try:
        while count < TARGET_SAMPLES:

            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            if line.startswith("timestamp_ms"):
                continue

            parts = line.split(",")

            # IMPORTANT: ESP32 sends 5 columns
            if len(parts) == 5:

                writer.writerow(parts)
                count += 1

                if count % 1000 == 0:
                    file.flush()
                    print(f"Recorded {count}/{TARGET_SAMPLES}")

            else:
                print("Skipped:", line)

        file.flush()
        print("\nTARGET REACHED!")
        print("CSV saved successfully.")

    except KeyboardInterrupt:
        file.flush()
        print("\nRecording stopped manually.")
        print("Samples recorded:", count)

ser.close()
