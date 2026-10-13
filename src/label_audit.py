
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "advanced_dataset.csv"


def count_runs(group):
    group = group.sort_values("time_s")
    labels = group["Label"].to_numpy(dtype=int)
    times = group["time_s"].to_numpy(dtype=float)

    positive = np.flatnonzero(labels == 1)
    if len(positive) == 0:
        return []

    runs = []
    start = previous = positive[0]

    for idx in positive[1:]:
        if idx == previous + 1 and times[idx] - times[previous] <= 2.5:
            previous = idx
        else:
            runs.append((start, previous))
            start = previous = idx

    runs.append((start, previous))

    return [
        {
            "Start_s": times[a],
            "End_s": times[b],
            "Duration_s": times[b] - times[a],
            "Samples": b - a + 1,
        }
        for a, b in runs
    ]


def main():
    df = pd.read_csv(DATA_PATH)
    df["Record"] = df["Record"].astype(str).str.zfill(2)

    print("LABEL AUDIT")
    print("===========")
    print("Rows:", len(df))
    print("Positive labels:", int(df["Label"].sum()))
    print("Positive rate: {:.3f}%".format(100 * df["Label"].mean()))

    print("\nSpO2 by label:")
    print(
        df.groupby("Label")["spo2"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(3)
        .to_string()
    )

    print("\nRecording-level summary:")
    rows = []

    for record, group in df.groupby("Record"):
        group = group.sort_values("time_s")
        runs = count_runs(group)

        rows.append({
            "Record": record,
            "Samples": len(group),
            "Duration_s": group["time_s"].max() - group["time_s"].min(),
            "Positive_Samples": int(group["Label"].sum()),
            "Positive_Rate_pct": 100 * group["Label"].mean(),
            "Min_SpO2": group["spo2"].min(),
            "Max_SpO2": group["spo2"].max(),
            "Positive_Runs": len(runs),
        })

    summary = pd.DataFrame(rows)
    print(summary.round(2).to_string(index=False))

    print("\nPositive runs:")
    for record, group in df.groupby("Record"):
        runs = count_runs(group)
        if not runs:
            continue

        print(f"\nRecord {record}: {len(runs)} run(s)")
        for run in runs:
            print(
                f"  {run['Start_s']:.1f}s–{run['End_s']:.1f}s; "
                f"{run['Duration_s']:.1f}s; "
                f"{run['Samples']} samples"
            )

    output = ROOT / "results" / "label_audit.csv"
    summary.to_csv(output, index=False)
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
