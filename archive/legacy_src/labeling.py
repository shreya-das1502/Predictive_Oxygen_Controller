
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

FEATURES_FILE = RESULTS_DIR / "features_record_01.csv"
OUTPUT_FILE = RESULTS_DIR / "labeled_record_01.csv"

# Experimental settings, not clinical thresholds
SPO2_THRESHOLD = 94.0
PREDICTION_HORIZON_SECONDS = 30.0

# ==========================================
# 2. LOAD FEATURES AND ORIGINAL SPO2
# ==========================================

if not FEATURES_FILE.exists():
    raise FileNotFoundError(
        f"Feature file not found: {FEATURES_FILE}. "
        "Run python src/features.py first."
    )

df = pd.read_csv(FEATURES_FILE)

# We need future SpO2 observations to create labels.
# Load the original numerical recording.
NUMERICS_FILE = PROJECT_DIR / "data" / "bidmc_01_Numerics.csv"

if not NUMERICS_FILE.exists():
    raise FileNotFoundError(f"Missing file: {NUMERICS_FILE}")

numerics = pd.read_csv(NUMERICS_FILE)
numerics.columns = numerics.columns.str.strip()

numerics["Time [s]"] = pd.to_numeric(
    numerics["Time [s]"], errors="coerce"
)
numerics["SpO2"] = pd.to_numeric(
    numerics["SpO2"], errors="coerce"
)

numerics = (
    numerics.dropna(subset=["Time [s]", "SpO2"])
    .sort_values("Time [s]")
    .reset_index(drop=True)
)

# ==========================================
# 3. LABEL FUTURE DESATURATION
# ==========================================

times = df["Time [s]"].to_numpy(dtype=float)
spo2_times = numerics["Time [s]"].to_numpy(dtype=float)
spo2_values = numerics["SpO2"].to_numpy(dtype=float)

labels = []
future_min_spo2 = []

for t in times:
    # Consider observations strictly after the current time,
    # up to 30 seconds into the future.
    mask = (
        (spo2_times > t)
        & (spo2_times <= t + PREDICTION_HORIZON_SECONDS)
    )

    future_values = spo2_values[mask]

    if len(future_values) == 0:
        labels.append(np.nan)
        future_min_spo2.append(np.nan)
    else:
        minimum = float(np.min(future_values))
        future_min_spo2.append(minimum)
        labels.append(int(minimum <= SPO2_THRESHOLD))

df["Future_Min_SpO2_30s"] = future_min_spo2
df["Desaturation_Within_30s"] = labels

# The final samples may lack a complete future window.
df = df.dropna(
    subset=["Future_Min_SpO2_30s", "Desaturation_Within_30s"]
).copy()

df["Desaturation_Within_30s"] = (
    df["Desaturation_Within_30s"].astype(int)
)

# ==========================================
# 4. REPORT LABEL DISTRIBUTION
# ==========================================

counts = df["Desaturation_Within_30s"].value_counts().sort_index()

print("\n========== LABELING COMPLETE ==========")
print("Rows:", len(df))
print("Prediction horizon:", PREDICTION_HORIZON_SECONDS, "seconds")
print("Experimental threshold:", SPO2_THRESHOLD, "%")
print("\nLabel counts:")
print(counts.rename(index={0: "No future low SpO2", 1: "Future low SpO2"}))

print("\nLabel proportions:")
print(df["Desaturation_Within_30s"].value_counts(normalize=True))

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved labeled dataset to:\n{OUTPUT_FILE}")
