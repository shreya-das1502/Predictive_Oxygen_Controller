
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path("results/elm_dataset.csv")
OUTPUT_DIR = Path("results")

FEATURES = [
    "PPG_Amplitude",
    "Pulse_Interval_s",
    "Pulse_Rate_BPM",
    "Amplitude_Slope",
]

TARGET_RECORD = "26"
COMPARISON_RECORDS = ["07", "12", "30"]


def normalize_record(value):
    return f"{int(float(value)):02d}"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run your dataset-building script first."
        )

    df = pd.read_csv(DATA_FILE)
    df["Record"] = df["Record"].apply(normalize_record)

    required = ["Record", "Label"] + FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required)

    # Recording-level summary.
    print("\n=== RECORDING SUMMARY ===")
    summary = (
        df.groupby("Record")
        .agg(
            Samples=("Label", "size"),
            Positives=("Label", "sum"),
            Positive_Rate=("Label", "mean"),
        )
    )
    summary["Positive_Rate"] *= 100
    print(summary.round(2).to_string())

    selected = [TARGET_RECORD] + COMPARISON_RECORDS
    subset = df[df["Record"].isin(selected)].copy()

    # Compare the feature distributions in Record 26 with
    # the other positive-containing test recordings.
    print("\n=== FEATURE DISTRIBUTIONS BY RECORD ===")
    stats = subset.groupby("Record")[FEATURES].agg(
        ["mean", "std", "median"]
    )
    print(stats.round(3).to_string())

    # Compare positive and negative labels within each record.
    print("\n=== POSITIVE VS NEGATIVE FEATURE MEANS ===")
    label_stats = (
        subset.groupby(["Record", "Label"])[FEATURES]
        .mean()
        .round(3)
    )
    print(label_stats.to_string())

    # Standardized mean differences: descriptive only.
    print("\n=== WITHIN-RECORD STANDARDIZED DIFFERENCES ===")
    for record in selected:
        part = subset[subset["Record"] == record]
        neg = part[part["Label"] == 0]
        pos = part[part["Label"] == 1]

        if len(neg) < 2 or len(pos) < 2:
            print(f"Record {record}: insufficient samples in a class")
            continue

        print(f"\nRecord {record}")
        for feature in FEATURES:
            pooled_sd = np.sqrt(
                (neg[feature].var() + pos[feature].var()) / 2
            )
            if not np.isfinite(pooled_sd) or pooled_sd == 0:
                difference = np.nan
            else:
                difference = (
                    pos[feature].mean() - neg[feature].mean()
                ) / pooled_sd

            print(f"  {feature}: {difference:.3f}")

    # Save summaries for easier inspection.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "recording_diagnostic_summary.csv")
    stats.to_csv(OUTPUT_DIR / "recording_diagnostic_features.csv")

    print("\nSaved:")
    print("  results/recording_diagnostic_summary.csv")
    print("  results/recording_diagnostic_features.csv")


if __name__ == "__main__":
    main()
