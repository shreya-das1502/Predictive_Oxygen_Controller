
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DATA_PATH = RESULTS / "advanced_dataset_cleaned.csv"
PREDICTIONS_PATH = RESULTS / "advanced_test_predictions.csv"

VALIDATION_RECORD = "12"

SCORE_COLUMNS = {
    "ELM": "ELM_Score",
    "Logistic Regression": "Logistic_Score",
    "Random Forest": "RF_Score",
}


def main():
    data = pd.read_csv(DATA_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    for frame in (data, predictions):
        frame["Record"] = (
            frame["Record"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(2)
        )

    validation = data[
        data["Record"] == VALIDATION_RECORD
    ].copy()

    print("\nVALIDATION LABEL DISTRIBUTION")
    print("=============================")
    print(validation["Label"].value_counts().sort_index())

    print("\nTEST SCORE DISTRIBUTIONS")
    print("========================")

    for model, column in SCORE_COLUMNS.items():
        if column not in predictions.columns:
            print(f"{model}: missing column {column}")
            continue

        scores = predictions[column].to_numpy()
        labels = predictions["Label"].to_numpy()

        print(f"\n{model}")
        print("  Score min:", round(float(np.min(scores)), 6))
        print("  Score max:", round(float(np.max(scores)), 6))
        print(
            "  Mean score for positives:",
            round(float(np.mean(scores[labels == 1])), 6)
            if np.any(labels == 1) else "N/A",
        )
        print(
            "  Mean score for negatives:",
            round(float(np.mean(scores[labels == 0])), 6)
            if np.any(labels == 0) else "N/A",
        )
        print(
            "  Test average precision:",
            round(float(average_precision_score(labels, scores)), 6)
            if len(np.unique(labels)) == 2 else "N/A",
        )

    print("\nTHRESHOLD DIAGNOSTIC ON TEST SCORES")
    print("===================================")
    print("Diagnostic only: do not use this to choose the final threshold.")

    for model, column in SCORE_COLUMNS.items():
        if column not in predictions.columns:
            continue

        scores = predictions[column].to_numpy()
        labels = predictions["Label"].to_numpy()

        print(f"\n{model}")
        print("Threshold | Precision | Recall | F1 | Predicted positives")

        for threshold in np.quantile(
            scores, [0.50, 0.75, 0.90, 0.95, 0.99]
        ):
            predicted = (scores >= threshold).astype(int)

            print(
                f"{threshold:9.5f} | "
                f"{precision_score(labels, predicted, zero_division=0):9.4f} | "
                f"{recall_score(labels, predicted, zero_division=0):6.4f} | "
                f"{f1_score(labels, predicted, zero_division=0):.4f} | "
                f"{int(predicted.sum())}"
            )


if __name__ == "__main__":
    main()
