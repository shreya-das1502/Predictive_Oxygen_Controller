
from pathlib import Path

import numpy as np
import pandas as pd

FILE = Path("results/recording_aware_elm_v2_predictions.csv")
OUTPUT = Path("results/elm_score_diagnostics.csv")


def main():
    if not FILE.exists():
        raise FileNotFoundError(
            f"{FILE} not found. Run recording_aware_elm_v2.py first."
        )

    df = pd.read_csv(FILE)

    required = {"Record", "Label", "Score", "Threshold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["Record"] = df["Record"].astype(str).str.zfill(2)

    rows = []

    for record, part in df.groupby("Record"):
        y = part["Label"].to_numpy(dtype=int)
        scores = part["Score"].to_numpy(dtype=float)
        threshold = float(part["Threshold"].iloc[0])

        pos = scores[y == 1]
        neg = scores[y == 0]

        print(f"\n{'=' * 45}")
        print(f"Record {record}")
        print(f"Samples: {len(part)}")
        print(f"Positive samples: {int(y.sum())}")
        print(f"Threshold: {threshold:.6f}")

        if len(pos):
            print(
                "Positive score percentiles (10, 50, 90):",
                np.round(np.percentile(pos, [10, 50, 90]), 6),
            )

        if len(neg):
            print(
                "Negative score percentiles (10, 50, 90):",
                np.round(np.percentile(neg, [10, 50, 90]), 6),
            )

        pred = scores >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fn = int(np.sum(~pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))

        # Show what happens at thresholds derived from this record's
        # score distribution. These are diagnostic only, NOT thresholds
        # to deploy: the held-out test labels must not tune the model.
        candidate_thresholds = (
            np.percentile(scores, [25, 50, 75, 90, 95])
            if len(scores) else []
        )

        print(f"Current TP={tp}, FN={fn}, FP={fp}")
        print("Diagnostic threshold sweep (not for model selection):")

        for candidate in candidate_thresholds:
            candidate_pred = scores >= candidate
            candidate_tp = int(np.sum(candidate_pred & (y == 1)))
            candidate_fn = int(np.sum(~candidate_pred & (y == 1)))
            candidate_fp = int(np.sum(candidate_pred & (y == 0)))

            recall = (
                candidate_tp / (candidate_tp + candidate_fn)
                if candidate_tp + candidate_fn else 0.0
            )

            print(
                f"  threshold={candidate:.6f} | "
                f"recall={recall:.3f} | "
                f"FN={candidate_fn} | FP={candidate_fp}"
            )

        rows.append({
            "Record": record,
            "Samples": len(part),
            "Positives": int(y.sum()),
            "Positive_Rate": float(y.mean()),
            "Threshold": threshold,
            "Positive_Score_Median": (
                float(np.median(pos)) if len(pos) else np.nan
            ),
            "Negative_Score_Median": (
                float(np.median(neg)) if len(neg) else np.nan
            ),
            "TP": tp,
            "FN": fn,
            "FP": fp,
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)
    print(f"\nSaved diagnostic summary to {OUTPUT}")


if __name__ == "__main__":
    main()
