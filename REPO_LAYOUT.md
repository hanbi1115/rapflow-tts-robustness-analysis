# Repository Layout

```text
rapflow-tts-robustness-analysis/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  └─ stress_test_sentences.csv
├─ scripts/
│  ├─ 01_speed_benchmark.py
│  ├─ 02_quality_eval.py
│  ├─ 03_visualize_baseline.py
│  ├─ 04_make_blind_test.py
│  ├─ 05_generate_stress_test.py
│  ├─ 06_analyze_stress_test.py
│  ├─ 07_visualize_results.py
│  └─ 08_make_blind_case_review.py
├─ results/
│  ├─ README.md
│  ├─ figures/
│  └─ tables/
└─ samples/
   └─ README.md
```

The pretrained RapFlow-TTS checkpoints, HiFi-GAN weights, and full generated audio set are intentionally excluded.
