
from pathlib import Path
import itertools
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
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("results/elm_dataset.csv")
OUTPUT_DIR = Path("results")

TARGET_RECALL = 0.92
RANDOM_SEED = 42

# These are the four sufficiently sized recordings containing
# positive labels in the dataset.
POSITIVE_RECORDS = ["07", "12", "26", "30"]

HIDDEN_NODES = [50, 100, 200]
RIDGE_VALUES = [0.1, 1.0]
POSITIVE_WEIGHTS = [1.0, 2.0, 5.0, 10.0]

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
# DATA PREPARATION
# ============================================================

def load_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. "
            "Run the existing dataset-building script first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = ["Record", "Label"] + BASE_FEATURES
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {INPUT_FILE}: {missing}"
        )

    # Normalize record IDs: 7, 7.0 and '07' become '07'.
    def normalize_record(value):
        value = str(value).strip()
        try:
            return f"{int(float(value)):02d}"
        except ValueError:
            return value.zfill(2)

    df["Record"] = df["Record"].apply(normalize_record)
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")

    for col in BASE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Record", "Label"] + BASE_FEATURES)
    df["Label"] = df["Label"].astype(int)

    # Causal rolling features: use current and past samples only.
    # Features are calculated separately for each recording.
    df = df.sort_values(["Record"]).copy()

    grouped = df.groupby("Record", sort=False)

    df["Rate_Mean_5"] = grouped["Pulse_Rate_BPM"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    df["Rate_Std_10"] = grouped["Pulse_Rate_BPM"].transform(
        lambda s: s.rolling(10, min_periods=2).std()
    )

    df["Rate_Change_5"] = grouped["Pulse_Rate_BPM"].transform(
        lambda s: s.diff(5)
    )

    df["Amplitude_Mean_5"] = grouped["PPG_Amplitude"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    df["Amplitude_Std_10"] = grouped["PPG_Amplitude"].transform(
        lambda s: s.rolling(10, min_periods=2).std()
    )

    df["Interval_Std_10"] = grouped["Pulse_Interval_s"].transform(
        lambda s: s.rolling(10, min_periods=2).std()
    )

    # Initial rolling statistics can be undefined. Keep the same
    # usable rows for every model and split.
    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    return df


# ============================================================
# EXTREME LEARNING MACHINE
# ============================================================

class ELM:
    def __init__(self, hidden_nodes=100, ridge=1.0,
                 positive_weight=2.0, seed=42):
        self.hidden_nodes = hidden_nodes
        self.ridge = ridge
        self.positive_weight = positive_weight
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)

        n_features = X.shape[1]

        self.input_weights = rng.normal(
            0, 1, size=(n_features, self.hidden_nodes)
        )
        self.hidden_bias = rng.normal(
            0, 1, size=(1, self.hidden_nodes)
        )

        H = np.tanh(X @ self.input_weights + self.hidden_bias)

        # Weighted ridge regression gives positive samples more
        # influence without duplicating validation/test samples.
        sample_weights = np.where(
            y == 1, self.positive_weight, 1.0
        )

        sqrt_weights = np.sqrt(sample_weights)
        Hw = H * sqrt_weights[:, None]
        yw = y * sqrt_weights

        identity = np.eye(self.hidden_nodes)

        self.output_weights = np.linalg.solve(
            Hw.T @ Hw + self.ridge * identity,
            Hw.T @ yw,
        )

        return self

    def predict_score(self, X):
        H = np.tanh(X @ self.input_weights + self.hidden_bias)
        return H @ self.output_weights


# ============================================================
# THRESHOLD SELECTION ON VALIDATION ONLY
# ============================================================

def choose_recall_first_threshold(y_true, scores):
    """
    Choose the highest threshold that reaches TARGET_RECALL.
    This minimizes predicted positives while satisfying the
    recall target on validation data.

    No test labels are used here.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    thresholds = np.unique(scores)
    thresholds = np.append(
        thresholds,
        np.min(scores) - 1e-10
    )

    candidates = []

    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)

        recall = recall_score(
            y_true, pred, zero_division=0
        )

        if recall >= TARGET_RECALL:
            tn, fp, fn, tp = confusion_matrix(
                y_true, pred, labels=[0, 1]
            ).ravel()

            candidates.append({
                "threshold": float(threshold),
                "recall": float(recall),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "tn": int(tn),
                "ap": float(average_precision_score(
                    y_true, scores
                )),
            })

    if not candidates:
        return None

    # Fewest false positives; then prefer higher threshold.
    candidates.sort(
        key=lambda x: (x["fp"], -x["threshold"])
    )

    return candidates[0]


# ============================================================
# METRICS
# ============================================================

def evaluate(y_true, scores, threshold):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Samples": len(y_true),
        "Positive_Samples": int(np.sum(y_true == 1)),
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
        "Average_Precision": average_precision_score(
            y_true, scores
        ),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


# ============================================================
# RECORDING-AWARE EXPERIMENT
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    print("\nPOSITIVE LABELS BY RECORD")
    print(
        df.groupby("Record")["Label"]
        .agg(["count", "sum"])
        .to_string()
    )

    available_positive_records = [
        record for record in POSITIVE_RECORDS
        if (
            record in set(df["Record"])
            and df.loc[df["Record"] == record, "Label"].sum() > 0
        )
    ]

    if len(available_positive_records) < 3:
        raise ValueError(
            "Need at least three positive-containing recordings "
            "to perform train/validation/test rotation."
        )

    fold_results = []
    prediction_results = []
    fold_number = 0

    # Each positive-containing recording becomes test once.
    # Each other positive-containing recording becomes validation.
    for test_record in available_positive_records:
        for validation_record in available_positive_records:
            if validation_record == test_record:
                continue

            fold_number += 1

            train_df = df[
                (~df["Record"].isin(
                    [test_record, validation_record]
                ))
            ].copy()

            val_df = df[
                df["Record"] == validation_record
            ].copy()

            test_df = df[
                df["Record"] == test_record
            ].copy()

            y_train = train_df["Label"].to_numpy()
            y_val = val_df["Label"].to_numpy()
            y_test = test_df["Label"].to_numpy()

            # Training must include both classes and positive
            # examples from recordings other than validation/test.
            if len(np.unique(y_train)) < 2 or np.sum(y_train) == 0:
                print(
                    f"Skipping fold: test={test_record}, "
                    f"validation={validation_record}; "
                    "training data lacks both classes."
                )
                continue

            if np.sum(y_val) == 0 or np.sum(y_test) == 0:
                print(
                    f"Skipping fold: test={test_record}, "
                    f"validation={validation_record}; "
                    "validation or test has no positives."
                )
                continue

            X_train_raw = train_df[FEATURES].to_numpy(dtype=float)
            X_val_raw = val_df[FEATURES].to_numpy(dtype=float)
            X_test_raw = test_df[FEATURES].to_numpy(dtype=float)

            # Fit preprocessing on training data only.
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)
            X_test = scaler.transform(X_test_raw)

            best_config = None
            best_model = None
            best_threshold_info = None

            # Select ELM settings and threshold using validation only.
            for hidden_nodes, ridge, pos_weight in itertools.product(
                HIDDEN_NODES, RIDGE_VALUES, POSITIVE_WEIGHTS
            ):
                model = ELM(
                    hidden_nodes=hidden_nodes,
                    ridge=ridge,
                    positive_weight=pos_weight,
                    seed=RANDOM_SEED,
                )

                model.fit(X_train, y_train)

                val_scores = model.predict_score(X_val)

                threshold_info = choose_recall_first_threshold(
                    y_val, val_scores
                )

                if threshold_info is None:
                    continue

                candidate_key = (
                    threshold_info["fp"],
                    -threshold_info["ap"],
                    -threshold_info["threshold"],
                )

                if (
                    best_config is None
                    or candidate_key < best_config["key"]
                ):
                    best_config = {
                        "key": candidate_key,
                        "hidden_nodes": hidden_nodes,
                        "ridge": ridge,
                        "positive_weight": pos_weight,
                    }
                    best_model = model
                    best_threshold_info = threshold_info

            if best_model is None:
                print(
                    f"No model selected for test={test_record}, "
                    f"validation={validation_record}."
                )
                continue

            # The held-out test recording is scored once with the
            # model and threshold selected without its labels.
            test_scores = best_model.predict_score(X_test)
            threshold = best_threshold_info["threshold"]

            metrics = evaluate(y_test, test_scores, threshold)

            metrics.update({
                "Fold": fold_number,
                "Test_Record": test_record,
                "Validation_Record": validation_record,
                "Train_Records": ",".join(
                    sorted(train_df["Record"].unique())
                ),
                "Validation_Recall": best_threshold_info["recall"],
                "Validation_FP": best_threshold_info["fp"],
                "Validation_FN": best_threshold_info["fn"],
                "Hidden_Nodes": best_config["hidden_nodes"],
                "Ridge": best_config["ridge"],
                "Positive_Weight": best_config["positive_weight"],
                "Validation_Target_Reached": (
                    best_threshold_info["recall"] >= TARGET_RECALL
                ),
                "Test_Target_Reached": (
                    metrics["Recall"] >= TARGET_RECALL
                ),
            })

            fold_results.append(metrics)

            test_output = test_df[
                ["Record", "Label"]
            ].copy()
            test_output["Fold"] = fold_number
            test_output["Validation_Record"] = validation_record
            test_output["Score"] = test_scores
            test_output["Threshold"] = threshold
            test_output["Prediction"] = (
                test_scores >= threshold
            ).astype(int)

            prediction_results.append(test_output)

            print(
                f"\nFold {fold_number}: "
                f"test={test_record}, validation={validation_record}"
            )
            print(
                f"  Train positive samples: {int(y_train.sum())}"
            )
            print(
                f"  Validation recall: "
                f"{best_threshold_info['recall']:.3f}, "
                f"validation FP: {best_threshold_info['fp']}"
            )
            print(
                f"  Test recall: {metrics['Recall']:.3f}, "
                f"test precision: {metrics['Precision']:.3f}, "
                f"test FN: {metrics['FN']}, "
                f"test FP: {metrics['FP']}"
            )

    if not fold_results:
        raise RuntimeError(
            "No folds completed. Check the dataset and label counts."
        )

    results_df = pd.DataFrame(fold_results)
    predictions_df = pd.concat(
        prediction_results, ignore_index=True
    )

    results_path = OUTPUT_DIR / "recording_aware_elm_results.csv"
    predictions_path = (
        OUTPUT_DIR / "recording_aware_elm_predictions.csv"
    )

    results_df.to_csv(results_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\n" + "=" * 65)
    print("RECORDING-AWARE ELM SUMMARY")
    print("=" * 65)

    summary = results_df.groupby("Test_Record").agg(
        Folds=("Fold", "count"),
        Mean_Recall=("Recall", "mean"),
        Min_Recall=("Recall", "min"),
        Mean_Precision=("Precision", "mean"),
        Mean_FN=("FN", "mean"),
        Mean_FP=("FP", "mean"),
        Mean_AP=("Average_Precision", "mean"),
        Target_Rate=("Test_Target_Reached", "mean"),
    )

    print(summary.round(3).to_string())

    print("\nOverall fold averages:")
    for metric in [
        "Recall", "Precision", "F1",
        "Average_Precision", "FN", "FP"
    ]:
        print(
            f"{metric}: {results_df[metric].mean():.4f}"
        )

    print(
        "\nTest folds reaching target recall: "
        f"{int(results_df['Test_Target_Reached'].sum())}"
        f"/{len(results_df)}"
    )

    print(f"\nSaved fold metrics to: {results_path}")
    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()
