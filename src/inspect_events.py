
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

record = 7
path = DATA / f"bidmc_{record:02d}_Numerics.csv"

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

events = []

for i, t in enumerate(times):
    if spo2[i] <= 94:
        continue

    start = np.searchsorted(times, t, side="right")
    end = np.searchsorted(times, t + 30, side="right")

    if start < end and np.any(spo2[start:end] <= 94):
        events.append({
            "Current_Time_s": t,
            "Current_SpO2": spo2[i],
            "Minimum_Future_SpO2": np.min(spo2[start:end]),
            "First_Future_Time_s": times[start],
            "Seconds_Until_Threshold": (
                times[start + np.where(spo2[start:end] <= 94)[0][0]] - t
            )
        })

result = pd.DataFrame(events)

print("\nRecord 07: Upcoming event inspection")
print("=" * 55)
print(f"Events found: {len(result)}")

if not result.empty:
    print("\nFirst 15 events:")
    print(result.head(15).to_string(index=False))

    print("\nSummary:")
    print(result.describe().round(2).to_string())

output = ROOT / "results"
output.mkdir(exist_ok=True)
result.to_csv(output / "record_07_events.csv", index=False)

print("\nSaved to results/record_07_events.csv")
python -u src/inspect_events.py