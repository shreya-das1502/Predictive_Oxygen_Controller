
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA_FILE = RESULTS / "elm_dataset.csv"

FEATURES = [
    "PPG_Amplitude",
    "Pulse_Interval_s",
    "Pulse_Rate_BPM",
    "Amplitude_Slope",
]

# --------------------------------------------------
# 1. Load existing dataset
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Run src/elm_experiment.py first to create "
        "results/elm_dataset.csv"
    )

df = pd.read_csv(DATA_FILE)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES + ["Target"])

X = df[FEATURES].to_numpy(dtype=float)
y = df["Target"].to_numpy(dtype=int)

print("Dataset shape:", df.shape)
print("Label counts:")
print(pd.Series(y).value_counts().sort_index())

if len(np.unique(y)) < 2:
    raise ValueError("Dataset must contain both classes.")

# Exploratory split, matching the previous experiment's approach.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

# Standardize using training data only.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# 2. Logistic Regression
# --------------------------------------------------

lr = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
)

start = time.perf_counter()
lr.fit(X_train, y_train)
lr_train_time = time.perf_counter() - start

start = time.perf_counter()
lr_scores = lr.predict_proba(X_test)[:, 1]
lr_inference_time = time.perf_counter() - start

# --------------------------------------------------
# 3. Extreme Learning Machine
# --------------------------------------------------

rng = np.random.default_rng(42)
hidden_neurons = 50

W = rng.normal(0, 1, (X_train.shape[1], hidden_neurons))
bias = rng.normal(size=hidden_neurons)

def hidden_layer(X):
    return np.tanh(X @ W + bias)

start = time.perf_counter()
H_train = hidden_layer(X_train)

# Balanced least-squares weights for the two classes.
counts = np.bincount(y_train, minlength=2)
sample_weights = np.array([
    len(y_train) / (2 * counts[label])
    for label in y_train
])

regularization = 0.01
A = H_train.T @ (sample_weights[:, None] * H_train)
A += regularization * np.eye(hidden_neurons)

b = H_train.T @ (sample_weights * y_train)
elm_weights = np.linalg.solve(A, b)

elm_train_time = time.perf_counter() - start

start = time.perf_counter()
elm_scores = hidden_layer(X_test) @ elm_weights
elm_inference_time = time.perf_counter() - start

# --------------------------------------------------
# 4. Compare metrics
# --------------------------------------------------

def evaluate(name, scores, train_time, inference_time):
    # Exploratory threshold only; choose final threshold on validation data.
    predictions = (scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Model": name,
        "Train_Time_s": train_time,
        "Inference_ms_per_sample": (
            inference_time / len(X_test) * 1000
        ),
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "PR_AUC_AP": average_precision_score(y_test, scores),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }

results = pd.DataFrame([
    evaluate("Logistic Regression", lr_scores,
             lr_train_time, lr_inference_time),
    evaluate("ELM", elm_scores,
             elm_train_time, elm_inference_time),
])

print("\nTest label counts:")
print(pd.Series(y_test).value_counts().sort_index())

print("\nMODEL COMPARISON")
print(results.round(5).to_string(index=False))

results.to_csv(RESULTS / "model_comparison.csv", index=False)
print("\nSaved to results/model_comparison.csv")