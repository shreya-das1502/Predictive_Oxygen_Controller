print("FEATURES SCRIPT STARTED")
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks

# ==========================================
# 1. PROJECT PATHS
# ==========================================

# features.py is inside src/
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIGNALS_FILE = DATA_DIR / "bidmc_01_Signals.csv"
NUMERICS_FILE = DATA_DIR / "bidmc_01_Numerics.csv"

OUTPUT_FILE = RESULTS_DIR / "features_record_01.csv"


# ==========================================
# 2. CHECK FILES
# ==========================================

if not SIGNALS_FILE.exists():
    raise FileNotFoundError(
        f"Signals file not found: {SIGNALS_FILE}"
    )

if not NUMERICS_FILE.exists():
    raise FileNotFoundError(
        f"Numerics file not found: {NUMERICS_FILE}"
    )


# ==========================================
# 3. LOAD DATA
# ==========================================

signals = pd.read_csv(SIGNALS_FILE)
numerics = pd.read_csv(NUMERICS_FILE)

# Clean column names
signals.columns = signals.columns.str.strip()
numerics.columns = numerics.columns.str.strip()

print("\nSignals columns:", signals.columns.tolist())
print("Numerics columns:", numerics.columns.tolist())

# Check required columns
required_signals = ["Time [s]", "PLETH"]
required_numerics = ["Time [s]", "SpO2"]

for col in required_signals:
    if col not in signals.columns:
        raise ValueError(f"Missing signal column: {col}")

for col in required_numerics:
    if col not in numerics.columns:
        raise ValueError(f"Missing numerics column: {col}")

# Convert to numeric and remove invalid rows
for col in required_signals:
    signals[col] = pd.to_numeric(
        signals[col], errors="coerce"
    )

for col in required_numerics:
    numerics[col] = pd.to_numeric(
        numerics[col], errors="coerce"
    )

signals = signals.dropna(
    subset=required_signals
).reset_index(drop=True)

numerics = numerics.dropna(
    subset=required_numerics
).sort_values("Time [s]").reset_index(drop=True)


# ==========================================
# 4. EXTRACT AND FILTER PPG
# ==========================================

time = signals["Time [s]"].to_numpy()
ppg = signals["PLETH"].to_numpy()

# BIDMC waveform sampling rate

time = signals["Time [s]"].to_numpy(dtype=float)
ppg = signals["PLETH"].to_numpy(dtype=float)

# Estimate sampling frequency from actual timestamps
fs = 1.0 / np.median(np.diff(time))

print(f"Using sampling frequency: {fs:.2f} Hz")


# Check the actual time spacing
if len(time) < 2:
    raise ValueError("Not enough PPG samples.")

actual_fs = 1.0 / np.median(np.diff(time))
print(f"\nEstimated sampling rate: {actual_fs:.2f} Hz")

if not np.isclose(actual_fs, fs, rtol=0.05):
    print("Warning: actual sampling rate differs from 125 Hz.")

# Fourth-order Butterworth bandpass filter
sos = butter(
    4,
    [0.5, 5.0],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered_ppg = sosfiltfilt(sos, ppg)


# ==========================================
# 5. DETECT PULSE PEAKS
# ==========================================

peaks, properties = find_peaks(
    filtered_ppg,
    distance=int(0.3 * fs),
    prominence=0.03
)

peak_times = time[peaks]
peak_values = filtered_ppg[peaks]

if len(peaks) < 2:
    raise ValueError(
        "Too few peaks detected. Check filter settings."
    )

print("Detected pulse peaks:", len(peaks))


# ==========================================
# 6. CALCULATE PULSE FEATURES
# ==========================================

intervals = np.diff(peak_times)

# Pulse amplitude relative to the lowest point
# between consecutive detected peaks
amplitudes = []

for i in range(len(peaks) - 1):
    segment = filtered_ppg[peaks[i]:peaks[i + 1]]

    amplitude = (
        peak_values[i] - np.min(segment)
    )

    amplitudes.append(amplitude)

amplitudes = np.asarray(amplitudes)

# Associate features with the earlier peak
feature_times = peak_times[:-1]

features = pd.DataFrame({
    "Time [s]": feature_times,
    "PPG_Amplitude": amplitudes,
    "Pulse_Interval_s": intervals
})

# Pulse rate estimate in beats per minute
features["Pulse_Rate_BPM"] = (
    60.0 / features["Pulse_Interval_s"]
)

# Recent pulse interval variability
features["Pulse_Interval_Variability"] = (
    features["Pulse_Interval_s"]
    .rolling(window=5, min_periods=2)
    .std()
)

# Amplitude change between adjacent beats
features["Amplitude_Slope"] = (
    features["PPG_Amplitude"].diff()
    / features["Pulse_Interval_s"]
)


# ==========================================
# 7. ESTIMATE PULSE RISE TIME
# ==========================================

rise_times = []

for peak in peaks[:-1]:
    start = max(0, peak - int(0.6 * fs))
    local_segment = filtered_ppg[start:peak + 1]

    minimum_index = start + np.argmin(local_segment)

    rise_time = time[peak] - time[minimum_index]
    rise_times.append(rise_time)

features["Rise_Time_s"] = rise_times


# ==========================================
# 8. ADD SPO2 FEATURES
# ==========================================

# Estimate SpO2 change over the previous five seconds
numerics["SpO2_Slope"] = (
    numerics["SpO2"] - numerics["SpO2"].shift(5)
) / (
    numerics["Time [s]"] - numerics["Time [s]"].shift(5)
)


# ==========================================
# ALIGN SPO2 FEATURES WITH PULSE FEATURES
# ==========================================

# Ensure both time columns use the same numeric type
features["Time [s]"] = pd.to_numeric(
    features["Time [s]"], errors="coerce"
).astype("float64")

numerics["Time [s]"] = pd.to_numeric(
    numerics["Time [s]"], errors="coerce"
).astype("float64")

numerics["SpO2"] = pd.to_numeric(
    numerics["SpO2"], errors="coerce"
)

# Remove invalid timestamps and sort before merging
features = (
    features.dropna(subset=["Time [s]"])
    .sort_values("Time [s]")
    .reset_index(drop=True)
)

spo2_data = (
    numerics[["Time [s]", "SpO2", "SpO2_Slope"]]
    .dropna(subset=["Time [s]"])
    .sort_values("Time [s]")
    .reset_index(drop=True)
)

# Match each pulse feature with the latest SpO2
# measurement at or before that timestamp
features = pd.merge_asof(
    features,
    spo2_data,
    on="Time [s]",
    direction="backward",
    tolerance=2.0
)

features = features.replace(
    [np.inf, -np.inf], np.nan
)


# ==========================================
# 9. SAVE FEATURE DATASET
# ==========================================

features.to_csv(OUTPUT_FILE, index=False)

print("\n========== FEATURE EXTRACTION COMPLETE ==========")
print("Feature rows:", len(features))
print("Feature columns:", features.columns.tolist())

print("\nFirst five rows:")
print(features.head())

print("\nMissing values:")
print(features.isna().sum())

print(f"\nSaved successfully to:\n{OUTPUT_FILE}")
