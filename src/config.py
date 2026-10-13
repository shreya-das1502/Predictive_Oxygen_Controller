"""Prespecified experiment settings; no outer-test-driven tuning."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
INPUT_DATASET = ROOT / 'results/elm_dataset.csv'
OUTPUT_DIR = ROOT / 'results/validated'
HORIZON = 30.0
SPO2_THRESHOLD = 94.0
MAX_NUMERIC_GAP = 1.01
SEED = 42
TARGET_RECALL = 0.92
INNER_FOLDS = 3
VERSION = 'causal-pulse-v1'
BASE_FEATURES = ['PPG_Amplitude', 'Pulse_Interval_s', 'Pulse_Rate_BPM', 'Amplitude_Slope']
ROLLING_FEATURES = ['Rate_Mean_5', 'Rate_Std_10', 'Rate_Change_5', 'Amplitude_Mean_5', 'Amplitude_Std_10', 'Interval_Std_10']
FEATURES = BASE_FEATURES + ROLLING_FEATURES
MODEL_CONFIGS = {
    'ELM': {'hidden_nodes': 50, 'ridge': 1.0, 'positive_weight': 5.0, 'seed': SEED},
    'Logistic': {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 2000, 'random_state': SEED},
    'GaussianNB': {'var_smoothing': 1e-6, 'priors': [0.5, 0.5]},
    'Tree': {'max_depth': 3, 'min_samples_leaf': 50, 'class_weight': 'balanced', 'random_state': SEED},
}
TIMING = {'warmup': 100, 'repetitions': 2000, 'timer': 'perf_counter_ns', 'threads': 1}
OUTPUT_FILES = ['dataset_audit.csv', 'label_audit.csv', 'per_recording_metrics.csv', 'pooled_metrics.csv', 'model_comparison.csv', 'per_record_predictions.csv', 'threshold_analysis.csv', 'latency_benchmark.csv', 'event_metrics.csv', 'run_manifest.json']
