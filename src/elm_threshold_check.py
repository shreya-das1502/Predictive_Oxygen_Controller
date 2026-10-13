
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)

from elm_experiment import FEATURES, fit_elm, predict_elm


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "elm_dataset.csv"


def main():
    data = pd.read_csv(DATA_PATH)

    X = data[FEATURES].to_numpy(dtype=float)
    y = data["Label"].to_numpy(dtype=int)

    if data["Record"].nunique() < 3:
        raise ValueError("Too few recordings for this diagnostic.")

    # Diagnostic only: random splitting can leak related samples
    # from the same recording into different partitions.
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

    # Select threshold on validation data only.
    thresholds = np.unique(
        np.quantile(val_scores, np.linspace(0, 1, 101))
    )

    candidates = []

    for threshold in thresholds:
        val_pred = (val_scores >= threshold).astype(int)

        precision = precision_score(
            y_val, val_pred, zero_division=0
        )
        recall = recall_score(
            y_val, val_pred, zero_division=0
        )

        # Diagnostic selection criterion: prioritize recall,
        # then precision. This is not clinical threshold tuning.
        candidates.append(
            (recall, precision, float(threshold))
        )

    best_recall, best_precision, best_threshold = max(
        candidates, key=lambda item: (item[0], item[1])
    )

    print("Validation samples:", len(y_val))
    print("Validation positives:", int(y_val.sum()))
    print("Chosen raw-score threshold:", best_threshold)
    print("Validation precision:", best_precision)
    print("Validation recall:", best_recall)

    # Evaluate the chosen threshold on the untouched test partition.
    test_pred = (test_scores >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_pred, labels=[0, 1]
    ).ravel()

    print("\nTest-set results")
    print("----------------")
    print("Accuracy:", accuracy_score(y_test, test_pred))
    print(
        "Precision:",
        precision_score(y_test, test_pred, zero_division=0),
    )
    print(
        "Recall:",
        recall_score(y_test, test_pred, zero_division=0),
    )
    print(
        "Average precision:",
        average_precision_score(y_test, test_scores),
    )
    print("TP:", tp)
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)

    print(
        "\nWARNING: This is still a random-row diagnostic, "
        "not a recording-independent evaluation."
    )


if __name__ == "__main__":
    main()
