
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "results" / "elm_dataset.csv"
RESULTS_DIR = ROOT / "results"

RESULTS_PATH = RESULTS_DIR / "recording_aware_elm_v2_results.csv"
PREDICTIONS_PATH = (
    RESULTS_DIR / "recording_aware_elm_v2_predictions.csv"
)

CHECKPOINT_RESULTS = (
    RESULTS_DIR / "recording_aware_elm_v2_results_checkpoint.csv"
)
CHECKPOINT_PREDICTIONS = (
    RESULTS_DIR / "recording_aware_elm_v2_predictions_checkpoint.csv"
)

TARGET_RECALL = 0.92
HIDDEN_NODES = 50
RIDGE = 1.0
POSITIVE_WEIGHT = 5.0
SEED = 42

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
LABEL = "Label"
RECORD = "Record"


# ============================================================
# DATA PREPARATION
# ============================================================

def load_dataset():
    print("Loading dataset...", flush=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = [RECORD, LABEL] + BASE_FEATURES
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["_source_row"] = np.arange(len(df))

    for col in BASE_FEATURES + [LABEL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[RECORD, LABEL] + BASE_FEATURES)
    df = df[df[LABEL].isin([0, 1])].copy()
    df[LABEL] = df[LABEL].astype(int)

    df = df.sort_values(
        [RECORD, "_source_row"]
    ).reset_index(drop=True)

    # Calculate causal rolling features independently per recording.
    # No future samples are used.
    grouped = df.groupby(RECORD, sort=False)

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

    # Retain initial rows when rolling history is unavailable.
    df[ROLLING_FEATURES] = (
        df[ROLLING_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Only remove rows with invalid essential features.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + [LABEL, RECORD]).copy()
    df[LABEL] = df[LABEL].astype(int)

    print(f"Loaded {len(df)} usable samples.", flush=True)
    print(f"Recordings: {df[RECORD].nunique()}", flush=True)
    print(f"Positive labels: {int(df[LABEL].sum())}", flush=True)

    print("\nSamples and positives per recording:")
    print(
        df.groupby(RECORD)[LABEL]
        .agg(Samples="count", Positives="sum")
        .to_string(),
        flush=True,
    )

    return df


# ============================================================
# EXTREME LEARNING MACHINE
# ============================================================

class ELM:
    def __init__(
        self,
        hidden_nodes=50,
        ridge=1.0,
        positive_weight=5.0,
        seed=42,
    ):
        self.hidden_nodes = hidden_nodes
        self.ridge = ridge
        self.positive_weight = positive_weight
        self.seed = seed

    def _hidden(self, X):
        return np.tanh(
            np.clip(
                X @ self.weights + self.bias,
                -30,
                30,
            )
        )

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)

        self.weights = rng.normal(
            0,
            1,
            size=(X.shape[1], self.hidden_nodes),
        )

        self.bias = rng.uniform(
            -1,
            1,
            size=(1, self.hidden_nodes),
        )

        H = self._hidden(X)

        sample_weights = np.where(
            y == 1,
            self.positive_weight,
            1.0,
        )

        sw = np.sqrt(sample_weights)
        Hw = H * sw[:, None]
        yw = y * sw

        matrix = (
            Hw.T @ Hw
            + self.ridge * np.eye(self.hidden_nodes)
        )

        target = Hw.T @ yw

        try:
            self.beta = np.linalg.solve(matrix, target)
        except np.linalg.LinAlgError:
            self.beta = np.linalg.pinv(matrix) @ target

        return self

    def score(self, X):
        return self._hidden(X) @ self.beta


def fit_model(train_df, seed=SEED):
    scaler = StandardScaler()

    X = scaler.fit_transform(
        train_df[FEATURES].to_numpy(dtype=float)
    )

    y = train_df[LABEL].to_numpy(dtype=int)

    model = ELM(
        hidden_nodes=HIDDEN_NODES,
        ridge=RIDGE,
        positive_weight=POSITIVE_WEIGHT,
        seed=seed,
    ).fit(X, y)

    return scaler, model


def get_scores(scaler, model, df):
    X = scaler.transform(
        df[FEATURES].to_numpy(dtype=float)
    )
    return model.score(X)


# ============================================================
# FAST THRESHOLD SELECTION
# ============================================================

def choose_threshold(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)

    valid = np.isfinite(scores)
    y = y[valid]
    scores = scores[valid]

    if len(y) == 0 or not np.any(y == 1):
        return float(np.nextafter(scores.min(), -np.inf))

    precision, recall, thresholds = precision_recall_curve(
        y,
        scores,
    )

    # precision and recall have one extra element.
    precision = precision[:-1]
    recall = recall[:-1]

    eligible = np.flatnonzero(recall >= TARGET_RECALL)

    if len(eligible):
        # Among thresholds reaching target recall, choose the one
        # with the best precision.
        idx = max(
            eligible,
            key=lambda i: (
                precision[i],
                recall[i],
                thresholds[i],
            ),
        )
    else:
        # If 92% recall is not achievable, maximize recall first.
        idx = max(
            range(len(thresholds)),
            key=lambda i: (
                recall[i],
                precision[i],
                thresholds[i],
            ),
        )

    return float(thresholds[idx])


# ============================================================
# FAST VALIDATION
# ============================================================

def select_threshold(train_df, test_record):
    """
    Use at most two recording-level validation fits:
    one positive-containing recording and one negative-only recording.

    Validation recordings are selected only from the training data.
    """

    record_ids = list(train_df[RECORD].drop_duplicates())

    positive_records = [
        r for r in record_ids
        if train_df.loc[train_df[RECORD] == r, LABEL].sum() > 0
    ]

    negative_records = [
        r for r in record_ids
        if train_df.loc[train_df[RECORD] == r, LABEL].sum() == 0
    ]

    if not positive_records:
        raise ValueError(
            f"No positive-containing validation recording available "
            f"while testing record {test_record}."
        )

    # Deterministic validation selection.
    validation_records = [positive_records[0]]

    if negative_records:
        validation_records.append(negative_records[0])

    validation_parts = []

    for i, val_record in enumerate(validation_records):
        fold_train = train_df[train_df[RECORD] != val_record]
        fold_val = train_df[train_df[RECORD] == val_record]

        if fold_train[LABEL].nunique() < 2:
            continue

        scaler, model = fit_model(
            fold_train,
            seed=SEED + i,
        )

        scores = get_scores(scaler, model, fold_val)

        validation_parts.append(
            pd.DataFrame({
                LABEL: fold_val[LABEL].to_numpy(),
                "Score": scores,
            })
        )

    if not validation_parts:
        raise RuntimeError(
            f"Validation failed for held-out record {test_record}."
        )

    validation = pd.concat(
        validation_parts,
        ignore_index=True,
    )

    return choose_threshold(
        validation[LABEL].to_numpy(),
        validation["Score"].to_numpy(),
    )


# ============================================================
# METRICS
# ============================================================

def evaluate(y, scores, threshold):
    y = np.asarray(y, dtype=int)
    prediction = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "Recall": recall_score(y, prediction, zero_division=0),
        "Precision": precision_score(y, prediction, zero_division=0),
        "F1": f1_score(y, prediction, zero_division=0),
        "Average_Precision": (
            average_precision_score(y, scores)
            if len(np.unique(y)) == 2
            else np.nan
        ),
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
        "Threshold": float(threshold),
        "Predicted_Positive": int(prediction.sum()),
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    records = sorted(
        df[RECORD].unique(),
        key=str,
    )

    results = []
    prediction_parts = []

    print("\nStarting fast recording-aware ELM evaluation.", flush=True)
    print(
        f"Fixed configuration: hidden={HIDDEN_NODES}, "
        f"ridge={RIDGE}, positive_weight={POSITIVE_WEIGHT}",
        flush=True,
    )

    for fold, test_record in enumerate(records, start=1):
        print(
            f"\n[{fold}/{len(records)}] "
            f"Testing recording {test_record}",
            flush=True,
        )

        test_df = df[df[RECORD] == test_record].copy()
        train_df = df[df[RECORD] != test_record].copy()

        print(
            f"Train={len(train_df)}, "
            f"Test={len(test_df)}, "
            f"Test positives={int(test_df[LABEL].sum())}",
            flush=True,
        )

        if train_df[LABEL].nunique() < 2:
            print("Skipping: training data contains only one class.", flush=True)
            continue

        # Threshold selection uses training recordings only.
        threshold = select_threshold(train_df, test_record)

        # Refit on all non-test recordings.
        scaler, model = fit_model(train_df, seed=SEED)

        scores = get_scores(scaler, model, test_df)

        metrics = evaluate(
            test_df[LABEL].to_numpy(),
            scores,
            threshold,
        )

        result = {
            "Test_Record": test_record,
            "Samples": len(test_df),
            "Positives": int(test_df[LABEL].sum()),
            **metrics,
            "Hidden_Nodes": HIDDEN_NODES,
            "Ridge": RIDGE,
            "Positive_Weight": POSITIVE_WEIGHT,
            "Test_Target_Reached": (
                metrics["Recall"] >= TARGET_RECALL
                if int(test_df[LABEL].sum()) > 0
                else np.nan
                ),
        }

        results.append(result)

        prediction_parts.append(
            pd.DataFrame({
                "Record": test_df[RECORD].to_numpy(),
                "_source_row": test_df["_source_row"].to_numpy(),
                "Label": test_df[LABEL].to_numpy(),
                "Score": scores,
                "Threshold": threshold,
                "Prediction": (scores >= threshold).astype(int),
            })
        )

        # Save after every completed fold.
        pd.DataFrame(results).to_csv(
            CHECKPOINT_RESULTS,
            index=False,
        )

        pd.concat(
            prediction_parts,
            ignore_index=True,
        ).to_csv(
            CHECKPOINT_PREDICTIONS,
            index=False,
        )

        print(
            f"Recall={metrics['Recall']:.3f}, "
            f"Precision={metrics['Precision']:.3f}, "
            f"FN={metrics['FN']}, "
            f"FP={metrics['FP']}",
            flush=True,
        )

        print(
            f"Elapsed: {(time.time() - start_time) / 60:.1f} minutes",
            flush=True,
        )

    if not results:
        raise RuntimeError("No recordings were evaluated.")

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    results_df.to_csv(RESULTS_PATH, index=False)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    print("\n=== FINAL RESULTS ===", flush=True)

    print(
        results_df[
            [
                "Test_Record",
                "Samples",
                "Positives",
                "Recall",
                "Precision",
                "F1",
                "TP",
                "FN",
                "FP",
            ]
        ].to_string(index=False),
        flush=True,
    )

    print("\n=== SUMMARY ===", flush=True)
    print(f"Mean recall: {results_df['Recall'].mean():.4f}")
    print(f"Mean precision: {results_df['Precision'].mean():.4f}")
    print(f"Mean F1: {results_df['F1'].mean():.4f}")
    print(f"Total false negatives: {results_df['FN'].sum()}")
    print(f"Total false positives: {results_df['FP'].sum()}")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(
        f"Total runtime: {(time.time() - start_time) / 60:.1f} minutes",
        flush=True,
    )


if __name__ == "__main__":
    main()
