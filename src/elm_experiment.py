
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)

# ============================================================
# CONFIGURATION
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def normalize_column_name(name):
    """Normalize column names, including 'Time [s]'."""
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
    )


def find_column(df, candidates):
    """Find a column without depending on capitalization or spaces."""
    lookup = {normalize_column_name(c): c for c in df.columns}

    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in lookup:
            return lookup[key]

    raise ValueError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def time_to_seconds(series):
    """Convert numeric seconds or timestamp strings to relative seconds."""
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().mean() > 0.95:
        values = numeric.to_numpy(dtype=float)
        return values - values[0]

    td = pd.to_timedelta(series.astype(str), errors="coerce")

    if td.notna().mean() > 0.95:
        values = td.dt.total_seconds().to_numpy(dtype=float)
        return values - values[0]

    dt = pd.to_datetime(series, errors="coerce")

    if dt.notna().mean() > 0.95:
        values = dt.astype("int64").to_numpy(dtype=float) / 1e9
        return values - values[0]

    raise ValueError(
        f"Could not convert time column '{series.name}' to seconds."
    )


# ============================================================
# LOAD AND PROCESS ONE RECORDING
# ============================================================
def load_record(record_number):
    record = f"{record_number:02d}"

    signals_path = DATA_DIR / f"bidmc_{record}_Signals.csv"
    numerics_path = DATA_DIR / f"bidmc_{record}_Numerics.csv"

    if not signals_path.exists() or not numerics_path.exists():
        print(f"Record {record}: missing Signals or Numerics file.")
        return None

    signals = pd.read_csv(signals_path)
    numerics = pd.read_csv(numerics_path)

    # Actual BIDMC files commonly use the column name 'Time [s]'.
    sig_time_col = find_column(
        signals, ["Time [s]", "Time", "Time(s)"]
    )
    ppg_col = find_column(signals, ["PLETH", "PPG"])

    num_time_col = find_column(
        numerics, ["Time [s]", "Time", "Time(s)"]
    )
    spo2_col = find_column(
        numerics, ["SpO2", "SPO2", "SpO₂"]
    )

    signals["time_s"] = time_to_seconds(signals[sig_time_col])
    numerics["time_s"] = time_to_seconds(numerics[num_time_col])

    signals["ppg"] = pd.to_numeric(
        signals[ppg_col], errors="coerce"
    )
    numerics["spo2"] = pd.to_numeric(
        numerics[spo2_col], errors="coerce"
    )

    signals = (
        signals.dropna(subset=["time_s", "ppg"])
        .sort_values("time_s")
        .reset_index(drop=True)
    )
    numerics = (
        numerics.dropna(subset=["time_s", "spo2"])
        .sort_values("time_s")
        .reset_index(drop=True)
    )

    if len(signals) < 100 or len(numerics) < 2:
        print(f"Record {record}: insufficient usable data.")
        return None

    time_values = signals["time_s"].to_numpy(dtype=float)
    intervals = np.diff(time_values)
    intervals = intervals[
        np.isfinite(intervals) & (intervals > 0)
    ]

    if len(intervals) == 0:
        print(f"Record {record}: invalid signal timestamps.")
        return None

    fs = 1.0 / np.median(intervals)

    if not 10 <= fs <= 1000:
        print(
            f"Record {record}: unexpected sampling rate "
            f"{fs:.2f} Hz."
        )
        return None

    ppg = signals["ppg"].to_numpy(dtype=float)

    # Band-pass filter for pulse-related activity.
    low_cutoff = 0.5
    high_cutoff = min(5.0, fs * 0.45)

    sos = butter(
        3,
        [low_cutoff, high_cutoff],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    try:
        filtered = sosfiltfilt(sos, ppg)
    except ValueError as exc:
        print(f"Record {record}: filtering failed: {exc}")
        return None

    # Detect candidate pulse peaks.
    peaks, _ = find_peaks(
        filtered,
        distance=max(1, int(0.30 * fs)),
        prominence=max(float(np.std(filtered)) * 0.15, 1e-12),
    )

    if len(peaks) < 3:
        print(f"Record {record}: too few PPG peaks.")
        return None

    rows = []

    for i in range(1, len(peaks)):
        peak = peaks[i]
        previous_peak = peaks[i - 1]

        pulse_interval = (
            time_values[peak] - time_values[previous_peak]
        )

        # Reject implausible pulse intervals.
        if not 0.30 <= pulse_interval <= 2.0:
            continue

        segment = filtered[previous_peak:peak + 1]
        amplitude = float(filtered[peak] - np.min(segment))

        if i >= 2:
            older_peak = peaks[i - 2]
            previous_segment = filtered[
                older_peak:previous_peak + 1
            ]
            previous_amplitude = float(
                filtered[previous_peak]
                - np.min(previous_segment)
            )
        else:
            previous_amplitude = amplitude

        rows.append(
            {
                "time_s": float(time_values[peak]),
                "PPG_Amplitude": amplitude,
                "Pulse_Interval_s": float(pulse_interval),
                "Pulse_Rate_BPM": float(60.0 / pulse_interval),
                "Amplitude_Slope": float(
                    (amplitude - previous_amplitude)
                    / pulse_interval
                ),
                "Record": record,
            }
        )

    if not rows:
        print(f"Record {record}: no valid pulse features.")
        return None

    features = pd.DataFrame(rows).sort_values("time_s")

    # Align each pulse to the most recent SpO2 reading.
    spo2_data = numerics[["time_s", "spo2"]].sort_values("time_s")

    aligned = pd.merge_asof(
        features,
        spo2_data,
        on="time_s",
        direction="backward",
        tolerance=2.0,
    )

    aligned = aligned.dropna(subset=["spo2"]).copy()

    spo2_times = spo2_data["time_s"].to_numpy(dtype=float)
    spo2_values = spo2_data["spo2"].to_numpy(dtype=float)

    labels = []

    for t, current_spo2 in zip(
        aligned["time_s"].to_numpy(),
        aligned["spo2"].to_numpy(),
    ):
        # Only predict from a current reading above the threshold.
        if current_spo2 <= SPO2_THRESHOLD:
            labels.append(np.nan)
            continue

        start = np.searchsorted(spo2_times, t, side="right")
        end = np.searchsorted(
            spo2_times,
            t + FUTURE_SECONDS,
            side="right",
        )

        future_values = spo2_values[start:end]

        if len(future_values) == 0:
            labels.append(np.nan)
        else:
            labels.append(
                int(np.any(future_values <= SPO2_THRESHOLD))
            )

    aligned["Label"] = labels
    aligned = aligned.dropna(subset=["Label"]).copy()
    aligned["Label"] = aligned["Label"].astype(int)

    return aligned[
        ["Record", "time_s", *FEATURES, "spo2", "Label"]
    ]


# ============================================================
# EXTREME LEARNING MACHINE
# ============================================================
def fit_elm(X_train, y_train, hidden_nodes=100, ridge=1e-2):
    """Fit a single-hidden-layer, ridge-regularized ELM."""
    rng = np.random.default_rng(42)

    input_weights = rng.normal(
        0, 1, size=(X_train.shape[1], hidden_nodes)
    )
    biases = rng.normal(0, 1, size=(1, hidden_nodes))

    hidden_output = np.tanh(
        X_train @ input_weights + biases
    )

    identity = np.eye(hidden_nodes)

    output_weights = np.linalg.solve(
        hidden_output.T @ hidden_output + ridge * identity,
        hidden_output.T @ y_train,
    )

    return input_weights, biases, output_weights


def predict_elm(X, model):
    """Return raw ELM scores; these are not calibrated probabilities."""
    input_weights, biases, output_weights = model

    hidden_output = np.tanh(X @ input_weights + biases)
    return hidden_output @ output_weights


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def main():
    frames = []

    print("Processing BIDMC records 01–30...\n")

    for record_number in range(FIRST_RECORD, LAST_RECORD + 1):
        try:
            frame = load_record(record_number)

            if frame is not None and not frame.empty:
                frames.append(frame)

                print(
                    f"Record {record_number:02d}: "
                    f"{len(frame)} samples, "
                    f"{int(frame['Label'].sum())} positive labels"
                )

        except Exception as exc:
            print(f"Record {record_number:02d} failed: {exc}")

    if not frames:
        raise RuntimeError(
            "No usable records found. Check the CSV filenames "
            "and column names."
        )

    data = pd.concat(frames, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURES + ["Label"]).copy()

    dataset_path = RESULTS_DIR / "elm_dataset.csv"
    data.to_csv(dataset_path, index=False)

    print("\nDataset saved:", dataset_path)
    print("Dataset shape:", data.shape)

    print("\nLabel counts:")
    print(data["Label"].value_counts().sort_index())

    print(
        f"\nPositive percentage: "
        f"{100 * data['Label'].mean():.3f}%"
    )

    print("\nSamples and positive labels by record:")
    print(
        data.groupby("Record")["Label"]
        .agg(Samples="count", Positives="sum")
        .to_string()
    )

    if data["Label"].nunique() < 2:
        print(
            "\nOnly one label class exists. Dataset saved, "
            "but classification evaluation cannot proceed."
        )
        return

    if data["Label"].value_counts().min() < 2:
        print(
            "\nToo few samples in one class for a stratified split."
        )
        return

    X = data[FEATURES].to_numpy(dtype=float)
    y = data["Label"].to_numpy(dtype=int)

    # Initial diagnostic only. This is NOT the final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    start = time.perf_counter()

    model = fit_elm(X_train, y_train)
    scores = predict_elm(X_test, model)

    elapsed = time.perf_counter() - start

    # Fixed baseline threshold only; later tune on validation data.
    predictions = (scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "Training_and_test_inference_time_s": elapsed,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Average_Precision": average_precision_score(
            y_test, scores
        ),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Test_samples": int(len(y_test)),
    }

    metrics_path = RESULTS_DIR / "elm_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print("\nInitial ELM evaluation:")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nMetrics saved:", metrics_path)
    print(
        "\nNote: this random row split can leak correlated samples "
        "between train and test. We will use recording-level "
        "validation before reporting final performance."
    )


if __name__ == "__main__":
    main()
