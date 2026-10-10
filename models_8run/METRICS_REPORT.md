# Final V4.1 study | Model 1–5 training metrics

These metrics are sourced from the original completed-run CSV files. Values are **validation metrics**, not sealed-test performance.

| Model | Last logged epoch | Precision | Recall | mAP50 | mAP50–95 | F1 | F1 source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1: YOLO26n synthetic_only | 196 | 0.8198 | 0.5884 | 0.6905 | 0.4996 | 0.6851 | derived |
| 2: YOLO26n hybrid | 200 | 0.8563 | 0.6175 | 0.7260 | 0.5277 | 0.7176 | derived |
| 3: YOLO26n-P2 synthetic_only | 200 | 0.8125 | 0.5893 | 0.6891 | 0.4972 | 0.6831 | derived |
| 4: YOLO26n-P2 hybrid | 200 | 0.8369 | 0.6268 | 0.7282 | 0.5296 | 0.7168 | derived |
| 5: RF-DETR Nano synthetic_only | 200 | 0.8646 | 0.6464 | 0.7238 | 0.4917 | 0.7396 | recorded |


## Available visualizations

Each model's `results/charts/` contains SVG plots for **Precision/Recall/mAP**, **F1**, and **loss**. Model 5 additionally has a chart for EMA mAP50-95. Each model has `metrics_summary.json` and `f1_by_epoch.csv`. These are derived from the stored original CSVs and are reproducible. For Model 1–4 the F1 values are calculated from the logged aggregate Precision and Recall; they are NOT the Ultralytics F1-confidence sweep. **For Model 5, F1 is recorded directly by the trainer.**

**Model 1 data completeness:** this source `results.csv` only covers epoch 1–196. Its separate training completion receipt says the run ended at epoch 200, but missing rows are not invented. The final value in this report is epoch 196, not epoch 200.

**Model 5 indexing:** trainer CSV is zero-indexed at epochs 150–199, displayed here as completed epochs 151–200; earlier epochs are not supplied by this CSV.

**Confusion matrices / official PR–F1-confidence curves:** These require saved validation predictions or original plot artifacts. They cannot be computed faithfully from the aggregate CSV alone. Do not mistake the derived per-epoch F1 line for the original confidence-threshold F1 curve.

**Scope:** validation metrics are not real-world or sealed-test performance. The 5 models are trained under different architecture/data conditions; comparisons require proper controlled evaluation.

