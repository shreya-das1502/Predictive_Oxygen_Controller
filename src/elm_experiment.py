
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FEATURES = [
    "PPG_Amplitude",
    "Pulse_Interval_s",
    "Pulse_Rate_BPM",
    "Amplitude_Slope",
]

# --------------------------------------------------
# 1. Extract features and provisional labels
# --------------------------------------------------

def process_record(record):
    signal_path = DATA / f"bidmc_{record:02d}_Signals.csv"
    numeric_path = DATA / f"bidmc_{record:02d}_Numerics.csv"

    if not signal_path.exists() or not numeric_path.exists():
        print(f"Record {record:02d}: files missing, skipped")
        return pd.DataFrame()

    signals = pd.read_csv(signal_path)
    numerics = pd.read_csv(numeric_path)

    signals.columns = signals.columns.str.strip()
    numerics.columns = numerics.columns.str.strip()

    signals["Time [s]"] = pd.to_numeric(
        signals["Time [s]"], errors="coerce"
    )
    signals["PLETH"] = pd.to_numeric(
        signals["PLETH"], errors="coerce"
    )
    numerics["Time [s]"] = pd.to_numeric(
        numerics["Time [s]"], errors="coerce"
    )
    numerics["SpO2"] = pd.to_numeric(
        numerics["SpO2"], errors="coerce"
    )

    signals = (
        signals.dropna(subset=["Time [s]", "PLETH"])
        .sort_values("Time [s]")
        .drop_duplicates("Time [s]")
        .reset_index(drop=True)
    )
    numerics = (
        numerics.dropna(subset=["Time [s]", "SpO2"])
        .sort_values("Time [s]")
        .drop_duplicates("Time [s]")
        .reset_index(drop=True)
    )

    if len(signals) < 100 or len(numerics) < 2:
        print(f"Record {record:02d}: insufficient data")
        return pd.DataFrame()

    signal_time = signals["Time [s]"].to_numpy(dtype=float)
    ppg = signals["PLETH"].to_numpy(dtype=float)

    fs = 1.0 / np.median(np.diff(signal_time))
    if not np.isfinite(fs) or fs <= 10:
        print(f"Record {record:02d}: invalid sampling rate")
        return pd.DataFrame()

    # Bandpass-filter the PPG.
    sos = butter(
        3, [0.5, 5.0], btype="bandpass",
        fs=fs, output="sos"
    )
    filtered = sosfiltfilt(sos, ppg)

    peaks, _ = find_peaks(
        filtered,
        distance=max(1, int(0.3 * fs)),
        prominence=max(0.01, 0.1 * np.std(filtered))
    )

    if len(peaks) < 3:
        print(f"Record {record:02d}: insufficient PPG peaks")
        return pd.DataFrame()

    rows = []

    numeric_times = numerics["Time [s]"].to_numpy(dtype=float)
    spo2 = numerics["SpO2"].to_numpy(dtype=float)

    for j in range(1, len(peaks)):
        previous = peaks[j - 1]
        peak = peaks[j]

        interval = signal_time[peak] - signal_time[previous]
        if interval <= 0:
            continue

        # Estimate pulse amplitude from the preceding valley.
        valley = previous + np.argmin(filtered[previous:peak + 1])
        amplitude = filtered[peak] - filtered[valley]

        t = float(signal_time[peak])

        # Find the latest available SpO2 at or before this time.
        current_index = np.searchsorted(
            numeric_times, t, side="right"
        ) - 1

        if current_index < 0:
            continue

        current_spo2 = spo2[current_index]

        # Only predict a future threshold crossing when current
        # SpO2 is above 94%.
        if current_spo2 <= 94:
            continue

        future_start = np.searchsorted(
            numeric_times, t, side="right"
        )
        future_end = np.searchsorted(
            numeric_times, t + 30, side="right"
        )

        # Skip rows without observed future measurements.
        if future_start >= future_end:
            continue

        target = int(
            np.any(spo2[future_start:future_end] <= 94)
        )

        rows.append({
            "Record": record,
            "Time [s]": t,
            "PPG_Amplitude": amplitude,
            "Pulse_Interval_s": interval,
            "Pulse_Rate_BPM": 60.0 / interval,
            "Amplitude_Slope": (
                amplitude - (
                    filtered[previous] - filtered[valley]
                )
            ) / interval,
            "Current_SpO2": current_spo2,
            "Target": target,
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        print(
            f"Record {record:02d}: {len(result)} samples, "
            f"{int(result['Target'].sum())} positive labels"
        )

    return result


# --------------------------------------------------
# 2. Process all ten records
# --------------------------------------------------

frames = []

for record in range(1, 11):
    frame = process_record(record)
    if not frame.empty:
        frames.append(frame)

if not frames:
    raise RuntimeError("No usable records found. Check data/.")

data = pd.concat(frames, ignore_index=True)
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=FEATURES + ["Target"])

data.to_csv(RESULTS / "elm_dataset.csv", index=False)

print("\nCombined dataset:", data.shape)
print("Label counts:")
print(data["Target"].value_counts().sort_index())

# --------------------------------------------------
# 3. Exploratory ELM experiment
# --------------------------------------------------

# ELM requires examples of both classes.
if data["Target"].nunique() < 2:
    raise RuntimeError(
        "Only one target class exists. ELM cannot learn "
        "a binary decision from this dataset."
    )

# IMPORTANT: This random split is only a smoke test.
# Nearby samples can overlap in time, so the test score
# must NOT be treated as independent-record performance.
rng = np.random.default_rng(42)
train_parts = []
test_parts = []

for label in [0, 1]:
    indices = np.flatnonzero(
        data["Target"].to_numpy() == label
    )
    rng.shuffle(indices)

    if len(indices) < 2:
        raise RuntimeError(
            f"Too few examples for class {label} to split."
        )

    cut = int(0.7 * len(indices))
    cut = min(max(cut, 1), len(indices) - 1)

    train_parts.extend(indices[:cut])
    test_parts.extend(indices[cut:])

train = data.iloc[train_parts]
test = data.iloc[test_parts]

X_train = train[FEATURES].to_numpy(dtype=float)
X_test = test[FEATURES].to_numpy(dtype=float)
y_train = train["Target"].to_numpy(dtype=float)
y_test = test["Target"].to_numpy(dtype=int)

# Standardize using training data only.
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std < 1e-8] = 1.0

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# ELM: random fixed input weights; learn output weights.
hidden_neurons = 50
input_size = X_train.shape[1]

W = rng.normal(
    0, 1, size=(input_size, hidden_neurons)
)
bias = rng.normal(size=hidden_neurons)

def hidden_layer(X):
    return np.tanh(X @ W + bias)

start = time.perf_counter()
H_train = hidden_layer(X_train)

# Ridge-regularized least-squares output layer.
regularization = 1e-2
output_weights = np.linalg.solve(
    H_train.T @ H_train
    + regularization * np.eye(hidden_neurons),
    H_train.T @ y_train
)
training_seconds = time.perf_counter() - start

start = time.perf_counter()
scores = hidden_layer(X_test) @ output_weights
# Inspect ELM score distribution for each class
print("\nELM score analysis")
print("Positive test scores:")
print(scores[y_test == 1])

print("\nScore ranges:")
if np.any(y_test == 0):
    print("Negative:", scores[y_test == 0].min(),
          "to", scores[y_test == 0].max())

if np.any(y_test == 1):
    print("Positive:", scores[y_test == 1].min(),
          "to", scores[y_test == 1].max())

# Try lower thresholds for exploratory analysis
for threshold in [0.05, 0.10, 0.20, 0.30, 0.50]:
    trial_predictions = (scores >= threshold).astype(int)

    tp = np.sum((trial_predictions == 1) & (y_test == 1))
    fp = np.sum((trial_predictions == 1) & (y_test == 0))
    fn = np.sum((trial_predictions == 0) & (y_test == 1))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    print(
        f"Threshold={threshold:.2f} | "
        f"TP={tp} FP={fp} FN={fn} | "
        f"Precision={precision:.3f} Recall={recall:.3f}"
    )
predictions = (scores >= 0.5).astype(int)
inference_seconds = time.perf_counter() - start

# Basic metrics.
tp = int(np.sum((predictions == 1) & (y_test == 1)))
tn = int(np.sum((predictions == 0) & (y_test == 0)))
fp = int(np.sum((predictions == 1) & (y_test == 0)))
fn = int(np.sum((predictions == 0) & (y_test == 1)))

accuracy = (tp + tn) / len(y_test) if len(y_test) else 0
precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0

print("\n========== ELM RESULTS ==========")
print(f"Training samples: {len(train)}")
print(f"Test samples: {len(test)}")
print(f"Hidden neurons: {hidden_neurons}")
print(f"Training time: {training_seconds:.6f} seconds")
print(f"Total test inference time: {inference_seconds:.6f} seconds")
print(
    "Mean inference time: "
    f"{inference_seconds / len(test) * 1000:.4f} ms/sample"
)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

metrics = pd.DataFrame([{
    "Model": "ELM",
    "Hidden_Neurons": hidden_neurons,
    "Train_Samples": len(train),
    "Test_Samples": len(test),
    "Training_Time_s": training_seconds,
    "Mean_Inference_ms": inference_seconds / len(test) * 1000,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "TP": tp, "TN": tn, "FP": fp, "FN": fn,
}])

metrics.to_csv(RESULTS / "elm_metrics.csv", index=False)
print("\nSaved dataset: results/elm_dataset.csv")
print("Saved metrics: results/elm_metrics.csv")
