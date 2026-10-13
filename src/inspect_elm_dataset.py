
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "results" / "elm_dataset.csv"

if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET}\n"
        "Run python src/elm_experiment.py first."
    )

df = pd.read_csv(DATASET)

print("=" * 65)
print("DATASET SUMMARY")
print("=" * 65)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nOverall label distribution:")
print(df["Label"].value_counts().sort_index())
print(f"\nPositive percentage: {100 * df['Label'].mean():.3f}%")

print("\nRecord-level summary:")
summary = df.groupby("Record").agg(
    Samples=("Label", "size"),
    Positive=("Label", "sum"),
    Mean_SpO2=("spo2", "mean"),
    Min_SpO2=("spo2", "min"),
    Max_SpO2=("spo2", "max"),
)

summary["Positive_Percent"] = (
    100 * summary["Positive"] / summary["Samples"]
)

print(summary.round(2).to_string())

print("\nSpO2 distribution for positive and negative labels:")
print(
    df.groupby("Label")["spo2"]
    .agg(["count", "mean", "min", "max"])
    .round(2)
)

print("\nConsecutive positive runs by recording:")
for record, group in df.groupby("Record"):
    group = group.sort_values("time_s")
    labels = group["Label"].to_numpy()
    times = group["time_s"].to_numpy()

    positive_indices = (labels == 1).nonzero()[0]

    if len(positive_indices) == 0:
        continue

    runs = []
    start = previous = positive_indices[0]

    for idx in positive_indices[1:]:
        # Consecutive pulse samples are approximately 0.3–2 seconds apart.
        if idx == previous + 1 and times[idx] - times[previous] <= 2.5:
            previous = idx
        else:
            runs.append((start, previous))
            start = previous = idx

    runs.append((start, previous))

    print(f"\nRecord {record}: {len(runs)} positive run(s)")
    for start, end in runs[:10]:
        duration = times[end] - times[start]
        print(
            f"  {times[start]:.1f}s to {times[end]:.1f}s "
            f"(about {duration:.1f}s, {end - start + 1} samples)"
        )

    if len(runs) > 10:
        print(f"  ... plus {len(runs) - 10} additional runs")
