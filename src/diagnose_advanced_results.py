
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
PATH = ROOT / "results" / "advanced_test_predictions.csv"
METRICS_PATH = ROOT / "results" / "advanced_model_comparison.csv"


def main():
    if not PATH.exists():
        raise FileNotFoundError(f"Missing file: {PATH}")

    data = pd.read_csv(PATH)
    metrics = pd.read_csv(METRICS_PATH)

    print("SAVED TEST PREDICTIONS")
    print("======================")
    print("Shape:", data.shape)
    print("Columns:", data.columns.tolist())

    print("\nPositive-label rate by test recording:")
    print(
        data.groupby("Record")["Label"]
        .agg(["count", "sum", "mean"])
        .to_string()
    )

    model_columns = [
        ("ELM", "ELM_Score", "ELM_Prediction"),
        ("Logistic Regression", "Logistic_Score",
         "Logistic_Prediction"),
    ]

    for name, score_col, pred_col in model_columns:
        print(f"\n\n{name}")
        print("=" * len(name))

        scores = data[score_col].to_numpy(dtype=float)
        labels = data["Label"].to_numpy(dtype=int)
        predictions = data[pred_col].to_numpy(dtype=int)

        print("\nScore distribution by true label:")
        print(
            data.assign(Score=scores)
            .groupby("Label")["Score"]
            .agg(["count", "mean", "std", "min", "median", "max"])
            .round(5)
            .to_string()
        )

        print("\nPrediction counts:")
        print(pd.Series(predictions).value_counts().sort_index())

        print("\nAverage precision:", round(
            average_precision_score(labels, scores), 5
        ))

        print("\nPer-record score diagnostics:")
        for record, group in data.groupby("Record"):
            y = group["Label"].to_numpy(dtype=int)
            s = group[score_col].to_numpy(dtype=float)
            p = group[pred_col].to_numpy(dtype=int)

            ap = (
                average_precision_score(y, s)
                if len(np.unique(y)) == 2
                else float("nan")
            )

            print(
                f"Record {record}: "
                f"samples={len(group)}, "
                f"positives={int(y.sum())}, "
                f"positive predictions={int(p.sum())}, "
                f"mean score positive="
                f"{s[y == 1].mean() if np.any(y == 1) else float('nan'):.5f}, "
                f"mean score negative="
                f"{s[y == 0].mean() if np.any(y == 0) else float('nan'):.5f}, "
                f"AP={ap:.5f}"
            )

        # Diagnostic only: shows the trade-off at different thresholds.
        # Do NOT choose the final threshold from test data.
        print("\nTest-set threshold diagnostic (not for model selection):")
        for threshold in np.quantile(scores, [0, .25, .5, .75, .9, .95, .99, 1]):
            p = (scores >= threshold).astype(int)
            print(
                f"threshold={threshold:.5f}, "
                f"predicted positives={int(p.sum())}, "
                f"precision={precision_score(labels, p, zero_division=0):.4f}, "
                f"recall={recall_score(labels, p, zero_division=0):.4f}, "
                f"F1={f1_score(labels, p, zero_division=0):.4f}"
            )

    print("\nThresholds used in the saved experiment:")
    print(
        metrics[["Model", "Validation_Threshold"]]
        .to_string(index=False)
    )

    print(
        "\nReminder: test-set threshold diagnostics are for understanding "
        "the score distributions only. Final thresholds must be chosen "
        "using validation recordings."
    )


if __name__ == "__main__":
    main()
