
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path("results/elm_dataset.csv")
PRED_FILE = Path("results/recording_aware_elm_v2_predictions.csv")

BASE_FEATURES = [
    "PPG_Amplitude",
    "Pulse_Interval_s",
    "Pulse_Rate_BPM",
    "Amplitude_Slope",
]


def normalize_record(value):
    return f"{int(float(value)):02d}"


def main():
    data = pd.read_csv(DATA_FILE)
    pred = pd.read_csv(PRED_FILE)

    for df in (data, pred):
        df["Record"] = df["Record"].apply(normalize_record)

    # Reproduce the rolling-feature preparation from Version 2.
    data["_source_row"] = np.arange(len(data))

    for col in BASE_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.sort_values(
        ["Record", "_source_row"]
    ).reset_index(drop=True)

    grouped = data.groupby("Record", sort=False)

    data["Rate_Mean_5"] = grouped["Pulse_Rate_BPM"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    data["Rate_Std_10"] = grouped["Pulse_Rate_BPM"].transform(
        lambda s: s.rolling(10, min_periods=2).std()
    )
    data["Rate_Change_5"] = grouped["Pulse_Rate_BPM"].transform(
        lambda s: s.diff(5)
    )
    data["Amplitude_Mean_5"] = grouped["PPG_Amplitude"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    data["Amplitude_Std_10"] = grouped["PPG_Amplitude"].transform(
        lambda s: s.rolling(10, min_periods=2).std()
    )
    data["Interval_Std_10"] = grouped["Pulse_Interval_s"].transform(
        lambda s: s.rolling(10, min_periods=2).std()
    )

    all_features = BASE_FEATURES + [
        "Rate_Mean_5",
        "Rate_Std_10",
        "Rate_Change_5",
        "Amplitude_Mean_5",
        "Amplitude_Std_10",
        "Interval_Std_10",
    ]

    data[all_features] = data[all_features].replace(
        [np.inf, -np.inf], np.nan
    )

    print("=== ROWS WITH MISSING FEATURES ===")
    for record in ["07", "12", "26", "30"]:
        part = data[data["Record"] == record]
        bad = part[part[all_features].isna().any(axis=1)]

        print(
            f"Record {record}: {len(bad)} rows affected; "
            f"positive labels among them = {int(bad['Label'].sum())}"
        )

        if len(bad):
            print(
                bad[["_source_row", "Label"]]
                .head(10)
                .to_string(index=False)
            )

    print("\n=== PREDICTION LABEL COUNTS ===")
    print(
        pred.groupby("Record")["Label"]
        .agg(["count", "sum"])
        .to_string()
    )


if __name__ == "__main__":
    main()
