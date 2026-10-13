
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, sosfiltfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FIRST_RECORD = 1
LAST_RECORD = 30
FUTURE_SECONDS = 30
SPO2_THRESHOLD = 94

FEATURES = [
    "PPG_Amplitude",
    "Pulse_Interval_s",
    "Pulse_Rate_BPM",
    "Amplitude_Slope",
]


def find_column(df, candidates):
    """Find a column without depending on capitalization."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def time_to_seconds(series):
    """Convert common BIDMC time formats to seconds."""
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().mean() > 0.95:
        values = numeric.to_numpy(dtype=float)
        values = values - values[0]
        return values

    td = pd.to_timedelta(series.astype(str), errors="coerce")
    if td.notna().mean() > 0.95:
        values = td.dt.total_seconds().to_numpy(dtype=float)
        return values - values[0]

    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().mean() > 0.95:
        values = dt.astype("int64").to_numpy(dtype=float) / 1e9
        return values - values[0]

    raise ValueError(f"Could not parse time column: {series.name}")


def load_record(record_number):
    record = f"{record_number:02d}"
    signals_path = DATA_DIR / f"bidmc_{record}_Signals.csv"
    numerics_path = DATA_DIR / f"bidmc_{record}_Numerics.csv"

    if not signals_path.exists() or not numerics_path.exists():
        print(f"Skipping record {record}: one or both CSV files are missing.")
        return None

    signals = pd.read_csv(signals_path)
    numerics = pd.read_csv(numerics_path)

    sig_time_col = find_column(signals, ["Time"])
    ppg_col = find_column(signals, ["PLETH", "PPG"])

    num_time_col = find_column(numerics, ["Time"])
    spo2_col = find_column(numerics, ["SpO2", "SPO2", "SpO₂"])

    signals["time_s"] = time_to_seconds(signals[sig_time_col])
    numerics["time_s"] = time_to_seconds(numerics[num_time_col])

    signals["ppg"] = pd.to_numeric(signals[ppg_col], errors="coerce")
    numerics["spo2"] = pd.to_numeric(numerics[spo2_col], errors="coerce")

    signals = signals.dropna(subset=["time_s", "ppg"]).sort_values("time_s")
    numerics = numerics.dropna(subset=["time_s", "spo2"]).sort_values("time_s")

    if len(signals) < 100 or len(numerics) < 2:
        print(f"Skipping record {record}: insufficient usable data.")
        return None

    dt = np.diff(signals["time_s"].to_numpy())
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        print(f"Skipping record {record}: invalid timestamps.")
        return None

    fs = 1.0 / np.median(dt)
    if fs <= 10:
        print(f"Skipping record {record}: unexpected PPG sampling rate {fs:.2f} Hz.")
        return None

    ppg = signals["ppg"].to_numpy(dtype=float)

    # Band-pass filter for pulse-related activity.
    sos = butter(
        3, [0.5, min(5.0, fs * 0.45)],
        btype="bandpass", fs=fs, output="sos"
    )
    filtered = sosfiltfilt(sos, ppg)

    # A minimum separation of 0.3 s prevents implausibly close peaks.
    peaks, _ = find_peaks(
        filtered,
        distance=max(1, int(0.30 * fs)),
        prominence=max(np.std(filtered) * 0.15, 1e-12),
    )

    if len(peaks) < 3:
        print(f"Skipping record {record}: too few PPG peaks.")
        return None

    times = signals["time_s"].to_numpy()
    rows = []

    for i in range(1, len(peaks)):
        peak = peaks[i]
        previous_peak = peaks[i - 1]

        interval = times[peak] - times[previous_peak]
        if not 0.30 <= interval <= 2.0:
            continue

        # Estimate pulse amplitude relative to the local trough.
        left = max(0, previous_peak)
        segment = filtered[left:peak + 1]
        amplitude = filtered[peak] - np.min(segment)

        previous_left = max(0, peaks[i - 2]) if i >= 2 else left
        previous_segment = filtered[previous_left:previous_peak + 1]
        previous_amplitude = (
            filtered[previous_peak] - np.min(previous_segment)
        )

        rows.append({
            "time_s": times[peak],
            "PPG_Amplitude": float(amplitude),
            "Pulse_Interval_s": float(interval),
            "Pulse_Rate_BPM": float(60.0 / interval),
            "Amplitude_Slope": float(
                (amplitude - previous_amplitude) / interval
            ),
            "Record": record,
        })

    if not rows:
        return None

    features = pd.DataFrame(rows).sort_values("time_s")

    # Align each pulse with the most recent available SpO2 reading.
    spo2 = numerics[["time_s", "spo2"]].sort_values("time_s")
    aligned = pd.merge_asof(
        features,
        spo2,
        on="time_s",
        direction="backward",
        tolerance=2.0,
    )
    aligned = aligned.dropna(subset=["spo2"]).copy()

    # Target: current SpO2 is above the threshold, then SpO2 reaches
    # the threshold or below within the next FUTURE_SECONDS.
    spo2_times = spo2["time_s"].to_numpy()
    spo2_values = spo2["spo2"].to_numpy()
    labels = []

    for t, current_spo2 in zip(aligned["time_s"], aligned["spo2"]):
        if current_spo2 <= SPO2_THRESHOLD:
            labels.append(np.nan)
            continue

        start = np.searchsorted(spo2_times, t, side="right")
        end = np.searchsorted(
            spo2_times, t + FUTURE_SECONDS, side="right"
        )
        future = spo2_values[start:end]

        if len(future) == 0:
            labels.append(np.nan)
        else:
            labels.append(int(np.any(future <= SPO2_THRESHOLD)))

    aligned["Label"] = labels
    aligned = aligned.dropna(subset=["Label"])
    aligned["Label"] = aligned["Label"].astype(int)

    return aligned[["Record", "time_s", *FEATURES, "spo2", "Label"]]


def train_elm(X_train, y_train, X_test, hidden_nodes=100, ridge=1e-2):
    """Single-hidden-layer ELM with tanh activation."""
    rng = np.random.default_rng(42)
    weights = rng.normal(0, 1, (X_train.shape[1], hidden_nodes))
    biases = rng.normal(0, 1, (1, hidden_nodes))

    H_train = np.tanh(X_train @ weights + biases)
    H_test = np.tanh(X_test @ weights + biases)

    # Ridge-regularized least-squares output weights.
    identity = np.eye(hidden_nodes)
    beta = np.linalg.solve(
        H_train.T @ H_train + ridge * identity,
        H_train.T @ y_train,
    )
    return H_test @ beta


def main():
    frames = []

    for number in range(FIRST_RECORD, LAST_RECORD + 1):
        try:
            frame = load_record(number)
            if frame is not None and not frame.empty:
                frames.append(frame)
                print(
                    f"Record {number:02d}: {len(frame)} samples, "
                    f"{int(frame['Label'].sum())} positive labels"
                )
        except Exception as exc:
            print(f"Record {number:02d} failed: {exc}")

    if not frames:
        raise RuntimeError(
            "No usable records found. Check the CSV filenames and columns."
        )

    data = pd.concat(frames, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURES + ["Label"])

    dataset_path = RESULTS_DIR / "elm_dataset.csv"
    data.to_csv(dataset_path, index=False)

    print("\nDataset saved:", dataset_path)
    print("Dataset shape:", data.shape)
    print("\nLabel counts:")
    print(data["Label"].value_counts().sort_index())
    print(f"Positive percentage: {100 * data['Label'].mean():.3f}%")

    print("\nPositive labels by record:")
    print(data.groupby("Record")["Label"].agg(["count", "sum"]))

    # This is an initial diagnostic split, not final generalization evidence.
    if data["Label"].nunique() < 2 or data["Label"].value_counts().min() < 2:
        print("\nNot enough examples of both classes for a stratified split.")
        print("Dataset was saved, but model evaluation was skipped.")
        return

    X = data[FEATURES].to_numpy(dtype=float)
    y = data["Label"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    start = time.perf_counter()
    scores = train_elm(X_train, y_train, X_test)
    training_and_fit_time = time.perf_counter() - start

    # ELM outputs are raw regression scores, not calibrated probabilities.
    # A 0.5 cutoff is used only as a basic initial baseline.
    predictions = (scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "Train_and_fit_time_s": training_and_fit_time,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "Average_Precision": average_precision_score(y_test, scores),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Test_samples": len(y_test),
    }

    metrics_path = RESULTS_DIR / "elm_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print("\nInitial ELM evaluation:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nMetrics saved:", metrics_path)
    print(
        "\nIMPORTANT: Random row splitting can leak correlated samples "
        "between train and test. Use recording-level validation before "
        "reporting final performance."
    )


if __name__ == "__main__":
    main()
