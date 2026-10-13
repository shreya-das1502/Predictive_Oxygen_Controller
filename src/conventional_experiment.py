
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)

from elm_experiment import FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "elm_dataset.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    data = pd.read_csv(DATA_PATH)

    X = data[FEATURES].to_numpy(dtype=float)
    y = data["Label"].to_numpy(dtype=int)

    # Diagnostic split matching the ELM threshold experiment.
    # We will replace row-wise evaluation with recording-level
    # evaluation before finalizing the model comparison.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    # Select a threshold on validation data only.
    validation_scores = model.predict_proba(X_val)[:, 1]
    thresholds = np.unique(
        np.quantile(validation_scores, np.linspace(0, 1, 101))
    )

    candidates = []

    for threshold in thresholds:
        pred = (validation_scores >= threshold).astype(int)

        precision = precision_score(
            y_val, pred, zero_division=0
        )
        recall = recall_score(
            y_val, pred, zero_division=0
        )

        # Select the threshold that maximizes F1 on validation data.
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        candidates.append((f1, float(threshold)))

    _, chosen_threshold = max(candidates, key=lambda x: x[0])

    # Measure prediction time separately from training.
    start = time.perf_counter()
    test_scores = model.predict_proba(X_test)[:, 1]
    inference_total = time.perf_counter() - start

    predictions = (test_scores >= chosen_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "Model": "Logistic Regression",
        "Training_Time_ms": training_time * 1000,
        "Inference_Total_ms": inference_total * 1000,
        "Inference_Median_us_per_sample": (
            np.median(
                [
                    (time.perf_counter() * 1e6)
                    for _ in range(1)
                ]
            )
            if len(X_test) == 0
            else np.nan
        ),
        "Chosen_Threshold": chosen_threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Average_Precision": average_precision_score(
            y_test, test_scores
        ),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Test_Samples": len(y_test),
    }

    output_path = RESULTS_DIR / "logistic_metrics.csv"
    pd.DataFrame([metrics]).to_csv(output_path, index=False)

    print("\nLogistic Regression results")
    print("---------------------------")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved:", output_path)
    print(
        "\nNote: this random-row split is exploratory only. "
        "Recording-level evaluation is required."
    )


if __name__ == "__main__":
    main()
