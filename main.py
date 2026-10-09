
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks

# ==========================================
# 1. PATHS AND DATA LOADING
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

signals_file = DATA_DIR / "bidmc_01_Signals.csv"
numerics_file = DATA_DIR / "bidmc_01_Numerics.csv"

signals = pd.read_csv(signals_file)
numerics = pd.read_csv(numerics_file)

signals.columns = signals.columns.str.strip()
numerics.columns = numerics.columns.str.strip()

# Check and clean required signal columns
signals["Time [s]"] = pd.to_numeric(
    signals["Time [s]"], errors="coerce"
)
signals["PLETH"] = pd.to_numeric(
    signals["PLETH"], errors="coerce"
)

signals = signals.dropna(
    subset=["Time [s]", "PLETH"]
).reset_index(drop=True)

# ==========================================
# 2. EXTRACT PPG
# ==========================================

fs = 125.0  # Sampling frequency in Hz

time = signals["Time [s]"].to_numpy()
ppg = signals["PLETH"].to_numpy()

# ==========================================
# 3. BUTTERWORTH BANDPASS FILTER
# ==========================================

low_cutoff = 0.5
high_cutoff = 5.0

sos = butter(
    4,
    [low_cutoff, high_cutoff],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered_ppg = sosfiltfilt(sos, ppg)

# ==========================================
# 4. PLOT RAW VS FILTERED PPG
# ==========================================

# Show a 10-second section for easier comparison
start_time = 30
end_time = 40

segment = (time >= start_time) & (time <= end_time)

plt.figure(figsize=(12, 5))

plt.plot(
    time[segment],
    ppg[segment],
    label="Raw PPG",
    alpha=0.65
)

plt.plot(
    time[segment],
    filtered_ppg[segment],
    label="Filtered PPG",
    linewidth=1.5
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Raw vs Filtered PPG")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    results / "PPG_Filtered_Comparison.png",
    dpi=300
)
plt.show()
plt.close()

# ==========================================
# 5. DETECT PULSE PEAKS
# ==========================================

# Minimum peak distance: 0.3 seconds
# This allows at most about 200 peaks per minute.
peaks, properties = find_peaks(
    filtered_ppg,
    distance=int(0.3 * fs),
    prominence=0.03
)

plt.figure(figsize=(12, 5))

plt.plot(
    time[segment],
    filtered_ppg[segment],
    label="Filtered PPG"
)

visible_peaks = peaks[segment[peaks]]

plt.scatter(
    time[visible_peaks],
    filtered_ppg[visible_peaks],
    color="red",
    s=25,
    label="Detected peaks"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("PPG Pulse Peak Detection")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    results / "PPG_Peaks.png",
    dpi=300
)
plt.show()
plt.close()

# ==========================================
# 6. SAVE FILTERED SIGNAL
# ==========================================

output = pd.DataFrame({
    "Time [s]": time,
    "Raw_PPG": ppg,
    "Filtered_PPG": filtered_ppg
})

output.to_csv(
    results / "filtered_ppg.csv",
    index=False
)

# ==========================================
# 7. SUMMARY
# ==========================================

print("\n--- SIGNAL PROCESSING COMPLETE ---")
print("Original PPG samples:", len(ppg))
print("Detected peaks:", len(peaks))
print("Saved filtered signal to results/filtered_ppg.csv")
print("Saved comparison graph to results/PPG_Filtered_Comparison.png")
print("Saved peak graph to results/PPG_Peaks.png")
