
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elm_experiment import FEATURES, fit_elm, predict_elm

DATA_PATH = ROOT / "results" / "elm_dataset.csv"
OUTPUT_PATH = ROOT / "results" / "latency_benchmark.csv"


def benchmark_predict(function, X, warmup=100, repetitions=1000):
    """Measure single-sample latency after warm-up."""
    sample = X[:1]

    for _ in range(warmup):
        function(sample)

    timings_ms = []

    for _ in range(repetitions):
        start = time.perf_counter_ns()
        function(sample)
        end = time.perf_counter_ns()
        timings_ms.append((end - start) / 1e6)

    timings_ms = np.asarray(timings_ms)

    return {
        "Median_Inference_ms": float(np.median(timings_ms)),
        "P95_Inference_ms": float(np.percentile(timings_ms, 95)),
        "Mean_Inference_ms": float(np.mean(timings_ms)),
        "Throughput_samples_per_s": float(
            1000 / np.mean(timings_ms)
        ),
        "Repetitions": repetitions,
    }


def main():
    data = pd.read_csv(DATA_PATH)

    X = data[FEATURES].to_numpy(dtype=float)
    y = data["Label"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = []

    # Logistic Regression
    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    start = time.perf_counter()
    logistic.fit(X_train, y_train)
    logistic_train_ms = (time.perf_counter() - start) * 1000

    logistic_bench = benchmark_predict(
        lambda batch: logistic.predict_proba(batch)[:, 1],
        X_test,
    )

    results.append({
        "Model": "Logistic Regression",
        "Training_Time_ms": logistic_train_ms,
        **logistic_bench,
    })

    # ELM
    start = time.perf_counter()
    elm = fit_elm(X_train, y_train)
    elm_train_ms = (time.perf_counter() - start) * 1000

    elm_bench = benchmark_predict(
        lambda batch: predict_elm(batch, elm),
        X_test,
    )

    results.append({
        "Model": "ELM",
        "Training_Time_ms": elm_train_ms,
        **elm_bench,
    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nLatency benchmark")
    print("=================")
    print(results_df.to_string(index=False))
    print("\nSaved:", OUTPUT_PATH)
    print(
        "\nThese figures measure model prediction only. "
        "They exclude sensor acquisition, preprocessing, "
        "feature extraction, serial communication and dashboard rendering."
    )


if __name__ == "__main__":
    main()
