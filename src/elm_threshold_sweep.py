
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elm_experiment import FEATURES, fit_elm, predict_elm

DATA_PATH = ROOT / "results" / "elm_dataset.csv"
RESULTS_PATH = ROOT / "results" / "elm_threshold_sweep.csv"


def evaluate_threshold(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true, pred, zero_division=0
    )
    recall = recall_score(
        y_true, pred, zero_division=0
    )

    return {
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision,
        "Recall": recall,
        "F1": f1_score(y_true, pred, zero_division=0),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def main():
    data = pd.read_csv(DATA_PATH)

    X = data[FEATURES].to_numpy(dtype=float)
    y = data["Label"].to_numpy(dtype=int)

    # Diagnostic only: this random-row split can mix samples
    # from the same recording across train/validation/test.
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

    model = fit_elm(X_train, y_train)

    val_scores = predict_elm(X_val, model)
    test_scores = predict_elm(X_test, model)

    # Test a range of candidate thresholds using validation scores.
    thresholds = np.unique(
        np.quantile(val_scores, np.linspace(0, 1, 301))
    )

    validation_rows = [
        evaluate_threshold(y_val, val_scores, threshold)
        for threshold in thresholds
    ]
    validation = pd.DataFrame(validation_rows)

    # Option A: threshold that maximizes validation accuracy.
    best_accuracy_row = validation.loc[
        validation["Accuracy"].idxmax()
    ]

    # Option B: highest validation accuracy among thresholds
    # achieving at least 80% validation recall, if any exist.
    recall_candidates = validation[
        validation["Recall"] >= 0.80
    ]

    if not recall_candidates.empty:
        best_recall_row = recall_candidates.loc[
            recall_candidates["Accuracy"].idxmax()
        ]
    else:
        best_recall_row = None

    print("\nValidation threshold selection")
    print("==============================")
    print("\nBest validation accuracy:")
    print(best_accuracy_row.to_string())

    if best_recall_row is not None:
        print("\nBest validation accuracy with recall >= 0.80:")
        print(best_recall_row.to_string())
    else:
        print("\nNo threshold achieved 80% validation recall.")

    # Thresholds are selected using validation data only.
    choices = [
        ("Max-validation-accuracy",
         float(best_accuracy_row["Threshold"]))
    ]

    if best_recall_row is not None:
        choices.append(
            ("Accuracy-with-recall-at-least-0.80",
             float(best_recall_row["Threshold"]))
        )

    test_rows = []

    print("\nUntouched test-set results")
    print("==========================")

    for name, threshold in choices:
        result = evaluate_threshold(
            y_test, test_scores, threshold
        )
        result["Selection_Rule"] = name
        result["Average_Precision"] = average_precision_score(
            y_test, test_scores
        )
        test_rows.append(result)

        print(f"\n{name}")
        for key, value in result.items():
            print(f"{key}: {value}")

    output = pd.DataFrame(test_rows)
    output.to_csv(RESULTS_PATH, index=False)
    print("\nSaved:", RESULTS_PATH)

    print(
        "\nIMPORTANT: This is still a random-row diagnostic. "
        "A high accuracy may result from predicting negatives. "
        "Recording-level validation is required for final claims."
    )


if __name__ == "__main__":
    main()
