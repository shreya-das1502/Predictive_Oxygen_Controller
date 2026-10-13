
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA_PATH = RESULTS / "elm_dataset.csv"

RANDOM_SEED = 42

THRESHOLD_SPO2 = 94.0
PREDICTION_HORIZON_SECONDS = 30.0

# Very short recordings are excluded from this experiment.
EXCLUDED_RECORDS = {"17", "29"}

# Record-level split. These sets must not overlap.
VALIDATION_RECORDS = {"12"}
TEST_RECORDS = {"07", "30"}

HIDDEN_NODES = 100
RIDGE = 0.1

BASE_FEATURES = [
    "PPG_Amplitude",
    "Pulse_Interval_s",
    "Pulse_Rate_BPM",
    "Amplitude_Slope",
]

ADVANCED_FEATURES = [
    "Rate_Mean_5",
    "Rate_Std_10",
    "Rate_Change_5",
    "Amplitude_Mean_5",
    "Amplitude_Std_10",
    "Interval_Std_10",
]


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def add_advanced_features(data):
    data = data.copy()

    data["Record"] = (
        data["Record"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )

    data = data.sort_values(
        ["Record", "time_s"]
    ).reset_index(drop=True)

    parts = []

    for record, group in data.groupby("Record", sort=False):
        group = group.copy()

        rate = group["Pulse_Rate_BPM"]
        amp = group["PPG_Amplitude"]
        interval = group["Pulse_Interval_s"]

        # Causal rolling features: current and past samples only.
        group["Rate_Mean_5"] = rate.rolling(
            5, min_periods=2
        ).mean()

        group["Rate_Std_10"] = rate.rolling(
            10, min_periods=3
        ).std()

        group["Rate_Change_5"] = rate - rate.shift(5)

        group["Amplitude_Mean_5"] = amp.rolling(
            5, min_periods=2
        ).mean()

        group["Amplitude_Std_10"] = amp.rolling(
            10, min_periods=3
        ).std()

        group["Interval_Std_10"] = interval.rolling(
            10, min_periods=3
        ).std()

        parts.append(group)

    result = pd.concat(parts, ignore_index=True)

    features = BASE_FEATURES + ADVANCED_FEATURES

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(
        subset=features + ["Label", "time_s"]
    ).copy()

    result["Label"] = result["Label"].astype(int)

    return result, features


# ============================================================
# 3. ELM
# ============================================================

def fit_elm_weighted(X, y, hidden_nodes=100, ridge=0.1):
    rng = np.random.default_rng(RANDOM_SEED)

    weights = rng.normal(
        0, 1, size=(X.shape[1], hidden_nodes)
    )
    biases = rng.normal(
        0, 1, size=(1, hidden_nodes)
    )

    hidden = np.tanh(X @ weights + biases)

    counts = np.bincount(y, minlength=2).astype(float)

    if np.any(counts == 0):
        raise ValueError(
            "ELM training data must contain both label classes."
        )

    # Balanced sample weighting for the rare positive class.
    sample_weights = len(y) / (2.0 * counts[y])

    weighted_hidden = (
        hidden * np.sqrt(sample_weights)[:, None]
    )
    weighted_y = y * np.sqrt(sample_weights)

    beta = np.linalg.solve(
        weighted_hidden.T @ weighted_hidden
        + ridge * np.eye(hidden_nodes),
        weighted_hidden.T @ weighted_y,
    )

    return weights, biases, beta


def predict_elm(X, model):
    weights, biases, beta = model
    return np.tanh(X @ weights + biases) @ beta


# ============================================================
# 4. METRICS AND THRESHOLD SELECTION
# ============================================================

def evaluate(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "Average_Precision": (
            average_precision_score(y_true, scores)
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def choose_threshold(y_val, scores):
    """
    Select a threshold using validation data only.
    Maximize F1; use recall and precision to break ties.
    """
    thresholds = np.unique(
        np.quantile(scores, np.linspace(0, 1, 301))
    )

    candidates = [
        evaluate(y_val, scores, threshold)
        for threshold in thresholds
    ]

    best = max(
        candidates,
        key=lambda row: (
            row["F1"],
            row["Recall"],
            row["Precision"],
        ),
    )

    return best["Threshold"], best


# ============================================================
# 5. LATENCY BENCHMARK
# ============================================================

def benchmark_inference(predict_function, X, repeats=5):
    """
    Measures model prediction latency on an already prepared
    feature matrix. Excludes file I/O and feature extraction.
    """
    # Warm-up
    predict_function(X[: min(20, len(X))])

    durations_ms = []

    for _ in range(repeats):
        start = time.perf_counter()
        predict_function(X)
        elapsed = time.perf_counter() - start

        durations_ms.append(elapsed * 1000)

    total_ms = float(np.median(durations_ms))
    per_sample_ms = total_ms / len(X)

    return {
        "Inference_Batch_Median_ms": total_ms,
        "Inference_Median_ms_per_sample": per_sample_ms,
        "Inference_P95_Batch_ms": float(
            np.percentile(durations_ms, 95)
        ),
        "Throughput_samples_per_second": (
            len(X) / (total_ms / 1000)
            if total_ms > 0 else np.nan
        ),
    }


# ============================================================
# 6. MAIN EXPERIMENT
# ============================================================

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_PATH}\n"
            "Run your dataset-generation script first."
        )

    original = pd.read_csv(DATA_PATH)

    required_columns = {
        "Record",
        "time_s",
        "Label",
        *BASE_FEATURES,
    }

    missing = required_columns - set(original.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    data, features = add_advanced_features(original)

    # Exclude only the specified short recordings.
    data = data[
        ~data["Record"].isin(EXCLUDED_RECORDS)
    ].copy()

    available_records = set(data["Record"].unique())

    validation_records = VALIDATION_RECORDS & available_records
    test_records = TEST_RECORDS & available_records

    if validation_records != VALIDATION_RECORDS:
        raise ValueError(
            "Validation recording(s) missing: "
            f"{sorted(VALIDATION_RECORDS - available_records)}"
        )

    if test_records != TEST_RECORDS:
        raise ValueError(
            "Test recording(s) missing: "
            f"{sorted(TEST_RECORDS - available_records)}"
        )

    if validation_records & test_records:
        raise ValueError(
            "Validation and test recordings must not overlap."
        )

    train_records = (
        available_records
        - validation_records
        - test_records
    )

    if not train_records:
        raise ValueError("No training recordings remain.")

    train = data[
        data["Record"].isin(train_records)
    ].copy()

    val = data[
        data["Record"].isin(validation_records)
    ].copy()

    test = data[
        data["Record"].isin(test_records)
    ].copy()

    print("\nRECORDING-LEVEL SPLIT")
    print("=====================")
    print("Training records:", sorted(train_records))
    print("Validation records:", sorted(validation_records))
    print("Test records:", sorted(test_records))
    print("Excluded records:", sorted(EXCLUDED_RECORDS))

    for name, frame in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        print(
            f"{name}: samples={len(frame)}, "
            f"positive={int(frame['Label'].sum())}, "
            f"negative={int((frame['Label'] == 0).sum())}, "
            f"recordings={frame['Record'].nunique()}"
        )

    if train["Label"].nunique() < 2:
        raise ValueError(
            "Training data needs both positive and negative labels."
        )

    if val["Label"].nunique() < 2:
        raise ValueError(
            "Validation data needs both classes to tune thresholds. "
            "Choose a validation recording with both labels."
        )

    if test["Label"].nunique() < 2:
        print(
            "WARNING: test data contains only one class; "
            "interpret some metrics cautiously."
        )

    X_train = train[features].to_numpy(dtype=float)
    y_train = train["Label"].to_numpy(dtype=int)

    X_val = val[features].to_numpy(dtype=float)
    y_val = val["Label"].to_numpy(dtype=int)

    X_test = test[features].to_numpy(dtype=float)
    y_test = test["Label"].to_numpy(dtype=int)

    # Scaling is fit on training data only.
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    results = []
    test_predictions = test[
        ["Record", "time_s", "Label"]
    ].copy()

    # --------------------------------------------------------
    # ELM
    # --------------------------------------------------------
    start = time.perf_counter()

    elm = fit_elm_weighted(
        X_train,
        y_train,
        hidden_nodes=HIDDEN_NODES,
        ridge=RIDGE,
    )

    elm_train_ms = (
        time.perf_counter() - start
    ) * 1000

    elm_val_scores = predict_elm(X_val, elm)
    elm_threshold, _ = choose_threshold(
        y_val, elm_val_scores
    )

    elm_test_scores = predict_elm(X_test, elm)
    elm_metrics = evaluate(
        y_test, elm_test_scores, elm_threshold
    )

    elm_latency = benchmark_inference(
        lambda X: predict_elm(X, elm),
        X_test,
    )

    results.append({
        "Model": "Weighted ELM",
        "Training_Time_ms": elm_train_ms,
        **elm_metrics,
        **elm_latency,
    })

    test_predictions["ELM_Score"] = elm_test_scores
    test_predictions["ELM_Prediction"] = (
        elm_test_scores >= elm_threshold
    ).astype(int)

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------
    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_SEED,
    )

    start = time.perf_counter()
    logistic.fit(X_train, y_train)
    logistic_train_ms = (
        time.perf_counter() - start
    ) * 1000

    logistic_val_scores = logistic.predict_proba(X_val)[:, 1]
    logistic_threshold, _ = choose_threshold(
        y_val, logistic_val_scores
    )

    logistic_test_scores = logistic.predict_proba(X_test)[:, 1]
    logistic_metrics = evaluate(
        y_test, logistic_test_scores, logistic_threshold
    )

    logistic_latency = benchmark_inference(
        lambda X: logistic.predict_proba(X)[:, 1],
        X_test,
    )

    results.append({
        "Model": "Logistic Regression",
        "Training_Time_ms": logistic_train_ms,
        **logistic_metrics,
        **logistic_latency,
    })

    test_predictions["Logistic_Score"] = logistic_test_scores
    test_predictions["Logistic_Prediction"] = (
        logistic_test_scores >= logistic_threshold
    ).astype(int)

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------
    forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    start = time.perf_counter()
    forest.fit(X_train, y_train)
    forest_train_ms = (
        time.perf_counter() - start
    ) * 1000

    forest_val_scores = forest.predict_proba(X_val)[:, 1]
    forest_threshold, _ = choose_threshold(
        y_val, forest_val_scores
    )

    forest_test_scores = forest.predict_proba(X_test)[:, 1]
    forest_metrics = evaluate(
        y_test, forest_test_scores, forest_threshold
    )

    forest_latency = benchmark_inference(
        lambda X: forest.predict_proba(X)[:, 1],
        X_test,
    )

    results.append({
        "Model": "Random Forest",
        "Training_Time_ms": forest_train_ms,
        **forest_metrics,
        **forest_latency,
    })

    test_predictions["RF_Score"] = forest_test_scores
    test_predictions["RF_Prediction"] = (
        forest_test_scores >= forest_threshold
    ).astype(int)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------
    metrics_path = RESULTS / "advanced_model_comparison.csv"
    predictions_path = RESULTS / "advanced_test_predictions.csv"
    dataset_path = RESULTS / "advanced_dataset_cleaned.csv"

    comparison = pd.DataFrame(results)

    comparison.to_csv(metrics_path, index=False)
    test_predictions.to_csv(predictions_path, index=False)
    data.to_csv(dataset_path, index=False)

    print("\nTEST-SET MODEL COMPARISON")
    print("=========================")

    display_columns = [
        "Model",
        "Training_Time_ms",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Average_Precision",
        "TP",
        "TN",
        "FP",
        "FN",
        "Inference_Median_ms_per_sample",
        "Inference_P95_Batch_ms",
        "Throughput_samples_per_second",
    ]

    print(
        comparison[display_columns]
        .round(5)
        .to_string(index=False)
    )

    print("\nSaved results:")
    print(metrics_path)
    print(predictions_path)
    print(dataset_path)

    print(
        "\nNOTE: Thresholds were selected using validation data only. "
        "The test set must not be used to tune thresholds or features. "
        "The inference benchmark measures model prediction on prepared "
        "features; it does not include sensor, preprocessing, or "
        "communication delays."
    )


if __name__ == "__main__":
    main()
