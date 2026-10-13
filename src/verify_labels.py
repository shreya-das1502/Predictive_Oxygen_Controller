
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DATASET_PATH = RESULTS_DIR / "advanced_dataset.csv"
OUTPUT_PATH = RESULTS_DIR / "label_verification.csv"

THRESHOLD = 94.0
HORIZON_SECONDS = 30.0


def load_numerics(record):
    path = DATA_DIR / f"bidmc_{record}_Numerics.csv"

    if not path.exists():
        return None

    numerics = pd.read_csv(path)
    numerics.columns = numerics.columns.str.strip()

    required = ["Time [s]", "SpO2"]
    if not all(column in numerics.columns for column in required):
        print(f"Skipping Record {record}: required columns missing.")
        return None

    numerics["Time [s]"] = pd.to_numeric(
        numerics["Time [s]"], errors="coerce"
    )
    numerics["SpO2"] = pd.to_numeric(
        numerics["SpO2"], errors="coerce"
    )

    numerics = (
        numerics.dropna(subset=required)
        .sort_values("Time [s]")
        .reset_index(drop=True)
    )

    return numerics


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)
    df["Record"] = (
        df["Record"].astype(str).str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )

    verification_parts = []

    print("VERIFYING FUTURE-DESATURATION LABELS")
    print("====================================")
    print(f"Threshold: SpO2 <= {THRESHOLD}%")
    print(f"Future horizon: {HORIZON_SECONDS} seconds")

    for record, group in df.groupby("Record", sort=True):
        group = group.copy()
        numerics = load_numerics(record)

        if numerics is None:
            print(f"Record {record}: source Numerics CSV unavailable.")
            continue

        times = group["time_s"].to_numpy(dtype=float)
        source_times = numerics["Time [s]"].to_numpy(dtype=float)
        source_spo2 = numerics["SpO2"].to_numpy(dtype=float)

        future_min = []
        expected_labels = []

        for t in times:
            start = np.searchsorted(source_times, t, side="right")
            end = np.searchsorted(
                source_times, t + HORIZON_SECONDS, side="right"
            )

            values = source_spo2[start:end]

            if len(values) == 0:
                future_min.append(np.nan)
                expected_labels.append(np.nan)
            else:
                minimum = float(np.min(values))
                future_min.append(minimum)
                expected_labels.append(
                    int(minimum <= THRESHOLD)
                )

        group["Recomputed_Future_Min_SpO2"] = future_min
        group["Expected_Label"] = expected_labels
        group["Existing_Label"] = group["Label"].astype(int)

        group = group.dropna(
            subset=["Expected_Label"]
        ).copy()
        group["Expected_Label"] = group[
            "Expected_Label"
        ].astype(int)

        group["Mismatch"] = (
            group["Existing_Label"] != group["Expected_Label"]
        )

        verification_parts.append(group)

        print(
            f"Record {record}: "
            f"samples={len(group)}, "
            f"existing positives={int(group['Existing_Label'].sum())}, "
            f"recomputed positives={int(group['Expected_Label'].sum())}, "
            f"mismatches={int(group['Mismatch'].sum())}"
        )

    if not verification_parts:
        raise RuntimeError(
            "No records could be verified. Check filenames and columns."
        )

    verified = pd.concat(verification_parts, ignore_index=True)
    verified.to_csv(OUTPUT_PATH, index=False)

    print("\nOVERALL")
    print("=======")
    print("Verified samples:", len(verified))
    print("Existing positives:", int(verified["Existing_Label"].sum()))
    print("Recomputed positives:", int(verified["Expected_Label"].sum()))
    print("Mismatches:", int(verified["Mismatch"].sum()))
    print(
        "Mismatch percentage: "
        f"{100 * verified['Mismatch'].mean():.3f}%"
    )

    print("\nRecords 17 and 29:")
    for record in ["17", "29"]:
        subset = verified[verified["Record"] == record]
        if len(subset):
            print(
                f"Record {record}: "
                f"existing positives={int(subset['Existing_Label'].sum())}, "
                f"recomputed positives={int(subset['Expected_Label'].sum())}, "
                f"mismatches={int(subset['Mismatch'].sum())}"
            )

    print("\nSaved detailed comparison to:", OUTPUT_PATH)
    print(
        "\nThis script verifies the target against the original Numerics "
        "files. It does not modify your training dataset."
    )


if __name__ == "__main__":
    main()
