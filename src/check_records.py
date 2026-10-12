
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

print("\nUPCOMING DESATURATION CHECK")
print("=" * 55)

for record in range(1, 11):
    path = DATA / f"bidmc_{record:02d}_Numerics.csv"

    if not path.exists():
        print(f"Record {record:02d}: Numerics file missing")
        continue

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df["Time [s]"] = pd.to_numeric(df["Time [s]"], errors="coerce")
    df["SpO2"] = pd.to_numeric(df["SpO2"], errors="coerce")
    df = (
        df.dropna(subset=["Time [s]", "SpO2"])
        .sort_values("Time [s]")
        .drop_duplicates("Time [s]")
        .reset_index(drop=True)
    )

    times = df["Time [s]"].to_numpy()
    spo2 = df["SpO2"].to_numpy()

    opportunities = 0
    positives = 0

    for i, t in enumerate(times):
        # Only predict when current SpO2 is above 94%.
        if spo2[i] <= 94:
            continue

        start = np.searchsorted(times, t, side="right")
        end = np.searchsorted(times, t + 30, side="right")

        if start >= end:
            continue

        opportunities += 1

        if np.any(spo2[start:end] <= 94):
            positives += 1

    negatives = opportunities - positives
    rate = 100 * positives / opportunities if opportunities else 0

    print(f"\nRecord {record:02d}")
    print(f"  Prediction opportunities: {opportunities}")
    print(f"  Upcoming events: {positives}")
    print(f"  No upcoming event: {negatives}")
    print(f"  Positive rate: {rate:.2f}%")

print("\nCheck complete.")
