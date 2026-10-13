
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA_PATH = RESULTS / "elm_dataset.csv"

VALIDATION_RECORDS = {"12"}
TEST_RECORDS = {"07", "30"}
EXCLUDED_RECORDS = {"17", "29"}

TARGET_RECALL = 0.92
RANDOM_SEED = 42

HIDDEN_NODE_OPTIONS = [50, 100, 200, 300]
RIDGE_OPTIONS = [0.01, 0.1, 1.0]
POSITIVE_WEIGHT_OPTIONS = [1.0, 2.0, 5.0, 10.0]

BASE_FEATURES = [
    "PPG_Amplitude",
    "Pulse_Interval_s",
    "Pulse_Rate_BPM",
    "Amplitude_Slope",
]

ROLLING_FEATURES = [
    "Rate_Mean_5",
    "Rate_Std_10",
    "Rate_Change_5",
    "Amplitude_Mean_5",
    "Amplitude_Std_10",
    "Interval_Std_10",
]

FEATURES = BASE_FEATURES + ROLLING_FEATURES


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(data):
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

    processed = []

    for _, group in data.groupby("Record", sort=False):
        group = group.copy()

        rate = group["Pulse_Rate_BPM"]
        amp = group["PPG_Amplitude"]
        interval = group["Pulse_Interval_s"]

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

        processed.append(group)

    data = pd.concat(processed, ignore_index=True)

    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.dropna(
        subset=FEATURES + ["Label", "time_s"]
    ).copy()

    data["Label"] = data["Label"].astype(int)

    return data


# ============================================================
# WEIGHTED ELM
# ============================================================

def train_elm(X, y, hidden_nodes, ridge, positive_multiplier):
    rng = np.random.default_rng(RANDOM_SEED)

    input_weights = rng.normal(
        0, 1, size=(X.shape[1], hidden_nodes)
    )
    biases = rng.normal(
        0, 1, size=(1, hidden_nodes)
    )

    hidden = np.tanh(X @ input_weights + biases)

    counts = np.bincount(y, minlength=2).astype(float)

    if np.any(counts == 0):
        raise ValueError(
            "ELM training requires both label classes."
        )

    # Balance classes, then allow additional positive weighting.
    sample_weights = len(y) / (2.0 * counts[y])
    sample_weights[y == 1] *= positive_multiplier

    sqrt_weights = np.sqrt(sample_weights)

    weighted_hidden = hidden * sqrt_weights[:, None]
    weighted_y = y.astype(float) * sqrt_weights

    regularized = (
        weighted_hidden.T @ weighted_hidden
        + ridge * np.eye(hidden_nodes)
    )

    output_weights = np.linalg.solve(
        regularized,
        weighted_hidden.T @ weighted_y,
    )

    return input_weights, biases, output_weights


def score_elm(X, model):
    input_weights, biases, output_weights = model

    hidden = np.tanh(X @ input_weights + biases)

    return hidden @ output_weights


# ============================================================
# METRICS AND RECALL-FIRST THRESHOLD
# ============================================================

def calculate_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    positives = int(np.sum(y_true == 1))

    return {
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "F1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "Average_Precision": float(
            average_precision_score(y_true, scores)
        ) if len(np.unique(y_true)) == 2 else np.nan,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Positive_Samples": positives,
    }


def choose_recall_first_threshold(y_true, scores, target_recall):
    """
    Select the highest threshold that meets the requested recall.

    This minimizes FP among the tested score thresholds that
    achieve the target recall on validation data.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    if not np.any(y_true == 1):
        raise ValueError("Validation set has no positive samples.")

    candidates = np.unique(scores)

    feasible = []

    for threshold in candidates:
        metrics = calculate_metrics(y_true, scores, threshold)

        if metrics["Recall"] >= target_recall:
            feasible.append(metrics)

    if not feasible:
        # The lowest score predicts every sample positive.
        threshold = float(np.nextafter(np.min(scores), -np.inf))
        metrics = calculate_metrics(y_true, scores, threshold)

        if metrics["Recall"] < target_recall:
            raise RuntimeError("Could not achieve target validation recall.")

        return threshold, metrics

    # Highest feasible threshold minimizes false positives.
    best = max(
        feasible,
        key=lambda row: (
            row["Threshold"],
            row["Precision"],
        ),
    )

    return best["Threshold"], best


# ============================================================
# MAIN
# ============================================================

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    raw = pd.read_csv(DATA_PATH)

    required = {
        "Record",
        "time_s",
        "Label",
        *BASE_FEATURES,
    }

    missing = required - set(raw.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = prepare_features(raw)

    data = data[
        ~data["Record"].isin(EXCLUDED_RECORDS)
    ].copy()

    available = set(data["Record"].unique())

    if not VALIDATION_RECORDS.issubset(available):
        raise ValueError(
            f"Missing validation records: "
            f"{sorted(VALIDATION_RECORDS - available)}"
        )

    if not TEST_RECORDS.issubset(available):
        raise ValueError(
            f"Missing test records: "
            f"{sorted(TEST_RECORDS - available)}"
        )

    if VALIDATION_RECORDS & TEST_RECORDS:
        raise ValueError("Validation and test recordings overlap.")

    train_records = available - VALIDATION_RECORDS - TEST_RECORDS

    train = data[data["Record"].isin(train_records)].copy()
    val = data[data["Record"].isin(VALIDATION_RECORDS)].copy()
    test = data[data["Record"].isin(TEST_RECORDS)].copy()

    if train["Label"].nunique() != 2:
        raise ValueError("Training data must contain both classes.")

    if val["Label"].nunique() != 2:
        raise ValueError("Validation data must contain both classes.")

    if test["Label"].nunique() != 2:
        raise ValueError("Test data must contain both classes.")

    print("\nRECALL-FIRST ELM EXPERIMENT")
    print("===========================")
    print("Training records:", sorted(train_records))
    print("Validation records:", sorted(VALIDATION_RECORDS))
    print("Test records:", sorted(TEST_RECORDS))
    print("Excluded records:", sorted(EXCLUDED_RECORDS))

    for name, frame in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        print(
            f"{name}: {len(frame)} samples, "
            f"{int(frame['Label'].sum())} positives, "
            f"{int((frame['Label'] == 0).sum())} negatives"
        )

    X_train_raw = train[FEATURES].to_numpy(dtype=float)
    y_train = train["Label"].to_numpy(dtype=int)

    X_val_raw = val[FEATURES].to_numpy(dtype=float)
    y_val = val["Label"].to_numpy(dtype=int)

    X_test_raw = test[FEATURES].to_numpy(dtype=float)
    y_test = test["Label"].to_numpy(dtype=int)

    # Fit scaling only on training recordings.
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # --------------------------------------------------------
    # Search model settings on validation data only.
    # --------------------------------------------------------

    candidates = []

    print("\nSEARCHING ELM CONFIGURATIONS")
    print("============================")

    for hidden_nodes in HIDDEN_NODE_OPTIONS:
        for ridge in RIDGE_OPTIONS:
            for positive_multiplier in POSITIVE_WEIGHT_OPTIONS:
                start = time.perf_counter()

                model = train_elm(
                    X_train,
                    y_train,
                    hidden_nodes,
                    ridge,
                    positive_multiplier,
                )

                train_ms = (time.perf_counter() - start) * 1000

                val_scores = score_elm(X_val, model)

                threshold, val_metrics = choose_recall_first_threshold(
                    y_val,
                    val_scores,
                    TARGET_RECALL,
                )

                row = {
                    "Hidden_Nodes": hidden_nodes,
                    "Ridge": ridge,
                    "Positive_Weight_Multiplier": positive_multiplier,
                    "Training_Time_ms": train_ms,
                    **{
                        f"Validation_{key}": value
                        for key, value in val_metrics.items()
                    },
                    "_model": model,
                    "_threshold": threshold,
                }

                candidates.append(row)

                print(
                    f"Nodes={hidden_nodes:3d}, "
                    f"ridge={ridge:g}, "
                    f"positive_weight={positive_multiplier:g}, "
                    f"recall={val_metrics['Recall']:.3f}, "
                    f"FP={val_metrics['FP']}, "
                    f"FN={val_metrics['FN']}"
                )

    # Prefer target recall, then fewer false positives, then higher AP.
    candidates.sort(
        key=lambda row: (
            row["Validation_Recall"] >= TARGET_RECALL,
            -row["Validation_FP"],
            row["Validation_Average_Precision"],
        ),
        reverse=True,
    )

    best = candidates[0]
    best_model = best["_model"]
    best_threshold = best["_threshold"]

    print("\nSELECTED USING VALIDATION DATA")
    print("==============================")
    print("Hidden nodes:", best["Hidden_Nodes"])
    print("Ridge:", best["Ridge"])
    print("Positive weight multiplier:", best["Positive_Weight_Multiplier"])
    print("Threshold:", best_threshold)
    print("Validation recall:", best["Validation_Recall"])
    print("Validation FP:", best["Validation_FP"])
    print("Validation FN:", best["Validation_FN"])

    # --------------------------------------------------------
    # Evaluate once on held-out test recordings.
    # --------------------------------------------------------

    test_scores = score_elm(X_test, best_model)

    test_metrics = calculate_metrics(
        y_test,
        test_scores,
        best_threshold,
    )

    print("\nUNSEEN TEST-SET RESULTS")
    print("=======================")

    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    if test_metrics["Recall"] >= TARGET_RECALL:
        print(
            f"\nTarget reached: test recall is at least "
            f"{TARGET_RECALL:.0%}."
        )
    else:
        print(
            f"\nTarget not reached: test recall is below "
            f"{TARGET_RECALL:.0%}. Improve training/features "
            "and evaluate again without tuning on this test set."
        )

    # Save the validation search without internal model objects.
    search_rows = []

    for row in candidates:
        search_rows.append({
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        })

    search_path = RESULTS / "recall_first_elm_search.csv"
    pd.DataFrame(search_rows).to_csv(search_path, index=False)

    metrics_path = RESULTS / "recall_first_elm_test_metrics.csv"
    pd.DataFrame([{
        "Model": "Recall-first ELM",
        "Target_Recall": TARGET_RECALL,
        "Hidden_Nodes": best["Hidden_Nodes"],
        "Ridge": best["Ridge"],
        "Positive_Weight_Multiplier": best[
            "Positive_Weight_Multiplier"
        ],
        "Validation_Threshold": best_threshold,
        **test_metrics,
    }]).to_csv(metrics_path, index=False)

    predictions_path = RESULTS / "recall_first_elm_test_predictions.csv"

    predictions = test[
        ["Record", "time_s", "Label"]
    ].copy()

    predictions["ELM_Score"] = test_scores
    predictions["ELM_Prediction"] = (
        test_scores >= best_threshold
    ).astype(int)

    predictions.to_csv(predictions_path, index=False)

    print("\nSaved:")
    print(search_path)
    print(metrics_path)
    print(predictions_path)

    print(
        "\nImportant: 92% is a target, not a guarantee. "
        "Threshold selection uses validation data only. "
        "Do not repeatedly tune against this test set."
    )


if __name__ == "__main__":
    main()
