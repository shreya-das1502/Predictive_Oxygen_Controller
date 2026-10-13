
from pathlib import Path

import pandas as pd

DATA_FILE = Path("results/elm_dataset.csv")
PRED_FILE = Path("results/recording_aware_elm_v2_predictions.csv")


def normalize_record(value):
    return f"{int(float(value)):02d}"


def main():
    for path in [DATA_FILE, PRED_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    data = pd.read_csv(DATA_FILE)
    pred = pd.read_csv(PRED_FILE)

    for df in [data, pred]:
        df["Record"] = df["Record"].apply(normalize_record)
        df["Label"] = pd.to_numeric(df["Label"], errors="raise").astype(int)

    print("\n=== CURRENT DATASET ===")
    print(
        data.groupby("Record").agg(
            Samples=("Label", "size"),
            Positives=("Label", "sum"),
        ).to_string()
    )

    print("\n=== SAVED TEST PREDICTIONS ===")
    print(
        pred.groupby("Record").agg(
            Samples=("Label", "size"),
            Positives=("Label", "sum"),
            Predicted_Positive=("Prediction", "sum"),
        ).to_string()
    )

    print("\n=== POSITIVE-COUNT COMPARISON ===")
    comparison = pd.concat(
        {
            "dataset": data.groupby("Record")["Label"].sum(),
            "predictions": pred.groupby("Record")["Label"].sum(),
        },
        axis=1,
    ).fillna(0).astype(int)

    print(comparison.to_string())

    print("\n=== PREDICTION FILE CHECKS ===")
    print("Prediction rows:", len(pred))
    print("Duplicate rows:", int(pred.duplicated().sum()))
    print("Missing scores:", int(pred["Score"].isna().sum()))
    print("Missing thresholds:", int(pred["Threshold"].isna().sum()))

    print(
        "\nNote: prediction files contain only held-out test records, "
        "not every dataset record."
    )


if __name__ == "__main__":
    main()
