
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "advanced_dataset.csv"

FEATURES = [
    "PPG_Amplitude",
    "Pulse_Interval_s",
    "Pulse_Rate_BPM",
    "Amplitude_Slope",
    "Rate_Mean_5",
    "Rate_Std_10",
    "Rate_Change_5",
    "Amplitude_Mean_5",
    "Amplitude_Std_10",
    "Interval_Std_10",
]


def main():
    data = pd.read_csv(DATA_PATH)

    print("FEATURE AUDIT")
    print("=============")
    print("Dataset shape:", data.shape)

    print("\nPositive labels by recording:")
    print(
        data.groupby("Record")["Label"]
        .agg(Samples="size", Positives="sum")
        .query("Positives > 0")
        .to_string()
    )

    available = [f for f in FEATURES if f in data.columns]
    missing = sorted(set(FEATURES) - set(available))

    if missing:
        print("\nFeatures not found:", missing)

    print("\nFeature comparison by label")
    print("===========================")

    rows = []

    for feature in available:
        negative = data.loc[data["Label"] == 0, feature].dropna()
        positive = data.loc[data["Label"] == 1, feature].dropna()

        if len(positive) < 2 or len(negative) < 2:
            continue

        neg_mean = negative.mean()
        pos_mean = positive.mean()
        neg_std = negative.std()
        pos_std = positive.std()

        pooled_std = np.sqrt(
            (neg_std**2 + pos_std**2) / 2
        )

        effect = (
            (pos_mean - neg_mean) / pooled_std
            if pooled_std > 0 else np.nan
        )

        correlation = data[[feature, "Label"]].corr().iloc[0, 1]

        rows.append({
            "Feature": feature,
            "Negative_Mean": neg_mean,
            "Positive_Mean": pos_mean,
            "Negative_Median": negative.median(),
            "Positive_Median": positive.median(),
            "Standardized_Mean_Difference": effect,
            "Label_Correlation": correlation,
        })

    report = pd.DataFrame(rows)
    report["Abs_Effect"] = report[
        "Standardized_Mean_Difference"
    ].abs()

    report = report.sort_values("Abs_Effect", ascending=False)

    print(report.drop(columns="Abs_Effect").round(4).to_string(index=False))

    output = ROOT / "results" / "feature_audit.csv"
    report.to_csv(output, index=False)
    print("\nSaved:", output)

    print(
        "\nInterpretation: large differences can indicate useful signal, "
        "but they do not prove predictive value. These comparisons are "
        "descriptive and do not replace held-out-recording evaluation."
    )


if __name__ == "__main__":
    main()
