# Measured computational latency

| Model | Component | Median_ms | P95_ms | P99_ms | Mean_ms | Max_ms | Repetitions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ELM | feature_calculation | 0.1144 | 0.2045 | 0.2532 | 0.1350 | 1.0188 | 2000 |
| ELM | model_only | 0.0055 | 0.0056 | 0.0064 | 0.0056 | 0.0302 | 2000 |
| ELM | transforms_and_model | 0.3272 | 0.5692 | 0.8903 | 0.3901 | 3.3960 | 2000 |
| ELM | thresholding | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0200 | 2000 |
| ELM | pulse_to_alert | 0.4954 | 0.7942 | 0.9757 | 0.5546 | 2.7170 | 2000 |
| ELM | waveform_sample_to_alert | 0.0709 | 0.1499 | 0.6194 | 0.0989 | 4.2017 | 19901 |
| ELM | waveform_pulse_producing_sample | 0.6308 | 1.1163 | undefined | 0.7535 | 3.1559 | 236 |
| GaussianNB | feature_calculation | 0.1130 | 0.1868 | 0.2233 | 0.1318 | 2.2151 | 2000 |
| GaussianNB | model_only | 0.2660 | 0.4282 | 0.6732 | 0.3013 | 2.4070 | 2000 |
| GaussianNB | transforms_and_model | 0.5719 | 1.0330 | 1.9807 | 0.7085 | 4.1199 | 2000 |
| GaussianNB | thresholding | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0008 | 2000 |
| GaussianNB | pulse_to_alert | 0.7370 | 1.2474 | 1.4224 | 0.8364 | 3.5006 | 2000 |
| GaussianNB | waveform_sample_to_alert | 0.0710 | 0.1429 | 0.8537 | 0.0991 | 3.2237 | 19901 |
| GaussianNB | waveform_pulse_producing_sample | 0.8833 | 1.4511 | undefined | 1.0154 | 3.2237 | 236 |
| Logistic | feature_calculation | 0.1137 | 0.1826 | 0.2116 | 0.1291 | 0.2602 | 2000 |
| Logistic | model_only | 0.1345 | 0.2188 | 0.2552 | 0.1551 | 0.4599 | 2000 |
| Logistic | transforms_and_model | 0.4587 | 0.7886 | 1.1794 | 0.5316 | 5.0310 | 2000 |
| Logistic | thresholding | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0207 | 2000 |
| Logistic | pulse_to_alert | 0.9043 | 1.8837 | 3.4978 | 1.0031 | 6.8583 | 2000 |
| Logistic | waveform_sample_to_alert | 0.0704 | 0.1354 | 0.7492 | 0.0927 | 2.8701 | 19901 |
| Logistic | waveform_pulse_producing_sample | 0.7708 | 1.2401 | undefined | 0.8486 | 2.1659 | 236 |
| Tree | feature_calculation | 0.1176 | 0.2428 | 0.3085 | 0.1493 | 2.8377 | 2000 |
| Tree | model_only | 0.1144 | 0.1624 | 0.1891 | 0.1249 | 0.4921 | 2000 |
| Tree | transforms_and_model | 0.4409 | 0.7566 | 0.9409 | 0.5067 | 3.2348 | 2000 |
| Tree | thresholding | 0.0002 | 0.0002 | 0.0003 | 0.0002 | 0.0008 | 2000 |
| Tree | pulse_to_alert | 0.6231 | 1.0315 | 1.4295 | 0.7071 | 3.9936 | 2000 |
| Tree | waveform_sample_to_alert | 0.0787 | 0.1664 | 0.7566 | 0.1104 | 5.5722 | 19901 |
| Tree | waveform_pulse_producing_sample | 0.7917 | 1.4462 | undefined | 0.9683 | 5.5722 | 236 |

Environment: {
  "python": "3.14.2",
  "platform": "Linux-6.8.0-1064-azure-x86_64-with-glibc2.39",
  "machine": "x86_64",
  "cpu": "AMD EPYC 7763 64-Core Processor",
  "logical_cpus": 2,
  "dependencies": {
    "numpy": "2.5.3",
    "pandas": "3.0.6",
    "scipy": "1.18.1",
    "sklearn": "1.9.1",
    "matplotlib": "3.11.2",
    "joblib": "1.6.0",
    "threadpoolctl": "3.7.0"
  },
  "threadpools": [
    {
      "user_api": "blas",
      "internal_api": "openblas",
      "num_threads": 1,
      "prefix": "libscipy_openblas",
      "filepath": "/usr/local/python/3.14.2/lib/python3.14/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so",
      "version": "0.3.34.106.0",
      "threading_layer": "pthreads",
      "architecture": "Haswell"
    },
    {
      "user_api": "openmp",
      "internal_api": "openmp",
      "num_threads": 1,
      "prefix": "libgomp",
      "filepath": "/usr/local/python/3.14.2/lib/python3.14/site-packages/scikit_learn.libs/libgomp-e985bcbb.so.1.0.0",
      "version": null
    },
    {
      "user_api": "blas",
      "internal_api": "openblas",
      "num_threads": 1,
      "prefix": "libscipy_openblas",
      "filepath": "/usr/local/python/3.14.2/lib/python3.14/site-packages/scipy.libs/libscipy_openblas-5f890258.so",
      "version": "0.3.31.dev",
      "threading_layer": "pthreads",
      "architecture": "Haswell"
    }
  ],
  "methodology": {
    "warmup": 100,
    "repetitions": 2000,
    "timer": "perf_counter_ns",
    "threads": 1
  },
  "measured_at": "2026-10-14T01:49:09.691048+05:30",
  "code_hashes": {
    "/workspaces/Predictive_Oxygen_Controller/src/__init__.py": "b191440a2acf4f15994559cce5ed9113c0e192b3d783e549a4f62ced31e81a50",
    "/workspaces/Predictive_Oxygen_Controller/src/advanced_experiment.py": "dbd3d43027529c36a286cbd5e12e3ae85fbd6938e5675777fd8cd36e956dcbea",
    "/workspaces/Predictive_Oxygen_Controller/src/benchmark_latency.py": "e410b218c8ef26dbe738256e9fa7beea848c6d9c0e49e92abb66e731fa4c3a91",
    "/workspaces/Predictive_Oxygen_Controller/src/check_dropped_samples.py": "66b533ce6caf6a5781d05ba90c1ca13abb865cc645df63c666abc7b8483d80e0",
    "/workspaces/Predictive_Oxygen_Controller/src/check_prediction_consistency.py": "89c3b6529b66d3dc8003c523dc385a66ba1c3915130a252cd95b83bc2db1d712",
    "/workspaces/Predictive_Oxygen_Controller/src/check_records.py": "77326de3b7118b95d4cc29e1f279f1c34cf587f5e4c413b774042d4fec22927c",
    "/workspaces/Predictive_Oxygen_Controller/src/compare_models.py": "0ef5dbd26ac484bca4019499c665ad69fc33ac4c0a993a712cb30d27ad1d7c09",
    "/workspaces/Predictive_Oxygen_Controller/src/config.py": "c4a1eab2369e17847ee2988d08f31b397865ad4320f9b6a8f0495f6b7dde2eb0",
    "/workspaces/Predictive_Oxygen_Controller/src/controller.py": "d91f4eaa838242d90ee548b221e07954f9ce4670da211a9fbd9133ecce6641f8",
    "/workspaces/Predictive_Oxygen_Controller/src/conventional_experiment.py": "4d71b3a69e79c1dd11169ddc09e5e95139fa8789112f0872e83087be30106ca3",
    "/workspaces/Predictive_Oxygen_Controller/src/dataset.py": "713f97feda5dcb8fcc85845ad32644875b0108d9d9a2cf651d62de7302d84711",
    "/workspaces/Predictive_Oxygen_Controller/src/diagnose_advanced_results.py": "2081e8a279d0a4d303191a5c33e8de6c1f2fb03bb267025a06a14d35fc27389a",
    "/workspaces/Predictive_Oxygen_Controller/src/diagnose_record_26.py": "d273155423c479028cebe40d8d2f4ffa471928db88e14a2ddfbdb8186d7df344",
    "/workspaces/Predictive_Oxygen_Controller/src/diagnose_thresholds.py": "3ea5188fe3486c199a8730449fd4bc06916bde5342464ced8d542da2c61f0aea",
    "/workspaces/Predictive_Oxygen_Controller/src/elm_experiment.py": "4da0b97ecd74820ac9ea95fa59d2b9d03eab657e9595526edb8c3c66476eddbf",
    "/workspaces/Predictive_Oxygen_Controller/src/elm_threshold_check.py": "08f4f8a4199d7ce6029c5d4e6c64ab68536600107821cedaabe5abb6ec5c187c",
    "/workspaces/Predictive_Oxygen_Controller/src/elm_threshold_sweep.py": "7f2683ab24c6b0abd8cbb00c4f7e06a847ca80ec6848e8c52c0e818175b6bc9d",
    "/workspaces/Predictive_Oxygen_Controller/src/evaluation.py": "fde8724f62e71f1ca073fea02a156bacdb7944429b79d1db71b3f5c2faf26841",
    "/workspaces/Predictive_Oxygen_Controller/src/feature_audit.py": "02c72ed305ecd218975efe0deb1474a1e4cb4e559eadab2e5c83a7167ed4a633",
    "/workspaces/Predictive_Oxygen_Controller/src/features.py": "cdcd865c481dd3534ec58c48c5f31d93f5eef98e40c32fad8a32fbae787c785b",
    "/workspaces/Predictive_Oxygen_Controller/src/inspect_elm_dataset.py": "01661777ea73a33949c8e0b691789bb3d74fee91fc84a478bd9db3a143480f66",
    "/workspaces/Predictive_Oxygen_Controller/src/inspect_elm_scores.py": "4c3861961a04afe50e9e0e29c7637829cc01815b2e44aa208d1c1aa8d10c4432",
    "/workspaces/Predictive_Oxygen_Controller/src/inspect_events.py": "95ac2ef7be146e7f6b66f5f1ea43ea466a22c132b13c5afd829e77c67b37482e",
    "/workspaces/Predictive_Oxygen_Controller/src/label_audit.py": "ae4a68c4d76f0097857b5232fa01df567e81d5159347fc43741377253328593d",
    "/workspaces/Predictive_Oxygen_Controller/src/labeling.py": "53ac0dfc229bd95386c0db1586870132920c0c01900ff798aab449d6f0c2606d",
    "/workspaces/Predictive_Oxygen_Controller/src/model.py": "c362baeeb93e16970ed54eaa090cf100f5177f790fa774cc97343679afd9f8c0",
    "/workspaces/Predictive_Oxygen_Controller/src/preprocessing.py": "7fcac04cdb70cfdeac5c22e057775c929269c6225e1b4c4f72f16fefeb67756a",
    "/workspaces/Predictive_Oxygen_Controller/src/recall_first_elm.py": "130cd239bf93e0e9bc10c3f59cd92afec07885783060e33a99868b5e9da9d0f5",
    "/workspaces/Predictive_Oxygen_Controller/src/recording_aware_elm.py": "731ab0bf458f5298a9937da998c35db8df6288506299fa37658064df9d61e439",
    "/workspaces/Predictive_Oxygen_Controller/src/recording_aware_elm_v2.py": "29bc9e58d59c992361bb295f30145169a35c8b2ee24705fd8a0653fe21cfbe59",
    "/workspaces/Predictive_Oxygen_Controller/src/replay.py": "7b02f79b9897aa399259ed0e21674269f4d92ce2f75615e5025cb35e75d5da2d",
    "/workspaces/Predictive_Oxygen_Controller/src/reporting.py": "8019f3a8f70b94e2bcda7cf764b15840421a88253d228f6ce90a2a7401d62822",
    "/workspaces/Predictive_Oxygen_Controller/src/run_experiments.py": "1c1f68f68bb8d3b68d5d961ed59c0a9cdac5fc7a15d9b4e4c74277f8a9437924",
    "/workspaces/Predictive_Oxygen_Controller/src/verify_labels.py": "ccb2a1fd20b4c1978d1b93962590eb5a47dabd6b4a7da5d114c36735d79afd75"
  }
}

Timer: perf_counter_ns; 100 warmups, 2,000 samples per pulse component, single native thread. Waveform replay times 19,901 post-warmup 125Hz inputs and separately reports pulse-producing inputs. No download, training, file writes, plots or loading occur in steady-state timers. Loading is measured once separately in CSV; it is not a percentile estimate. Component times are measured independently and should not simply be summed. P99 is reported only for at least 1,000 measurements; the smaller pulse-producing waveform subset has no reliable P99. P99 is empirical, not a worst-case bound; shared-host scheduling explains outliers. Throughput is reciprocal mean compute time, not sensor rate.

Waveform filtering is forward SOS with zero initial state. Local maxima require one following waveform sample (8ms confirmation); intervals/amplitudes use the preceding beat. Signals arrive at 125Hz, while alert opportunities occur per accepted pulse. Computing a pulse feature requires acquiring that pulse and its history (rolling windows are 5/10 pulses, not seconds). Filtering phase delay, sensor/monitor latency, communication and actuation are not measured. The 30s label horizon is neither computational latency nor guaranteed warning lead time. Research replay is not a real-time or clinical deployment guarantee.
