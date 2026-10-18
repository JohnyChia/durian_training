# Eight-Run Model Archive — Current Status

Branch: **`model-synthetic-and-hybrid`**  
Updated: **18 October 2026**  
Scope: **archived training runs, not a final common-test leaderboard**.

This folder currently contains **six completed 200-epoch training runs** from the intended four-architecture × two-condition study. Models 7 and 8 (D-FINE Nano) are **not present** in this archive.

## Run-by-run inventory

| Model | Run | Architecture | Training dataset | Main weights | Metric log completeness | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| [Model 1](model1_yolo26n_synthetic/) | 01 | YOLO26n | V4.1 Synthetic-only | `best.pt` | **E1–E196**; E200 completion noted separately | `results/metrics_summary.json` |
| [Model 2](model2_yolo26n_hybrid/) | 02 | YOLO26n | **V4.2 corrected Hybrid** | `best.pt`, `weights/last.pt`, saved intermediate weights | E1–E200 training CSV archived | `MODEL2_E200_MANIFEST.json`; [README](model2_yolo26n_hybrid/README.md) |
| [Model 3](model3_yolo26n_p2_synthetic/) | 03 | YOLO26n-P2 | V4.1 Synthetic-only | `best.pt` | **E1–E200** | `results/metrics_summary.json` |
| [Model 4](model4_yolo26n_p2_hybrid/) | 04 | YOLO26n-P2 | **V4.2 corrected Hybrid** | `best.pt`, `last.pt` | **E1–E200** merged/verified | [README](model4_yolo26n_p2_hybrid/README.md), `FINAL_AUDIT.json`, `ARCHIVE_MANIFEST.json` |
| [Model 5](model5_rfdetr_n_synthetic/) | 05 | RF-DETR Nano | V4.1 Synthetic-only | `weights/*.pth` and full `checkpoints/last.ckpt` | **E151–E200** only in published CSV | `MODEL5_MANIFEST.json`; `results/training_config.json` |
| [Model 6](model6_rfdetr_n_hybrid/) | 06 | RF-DETR Nano | **V4.2 corrected Hybrid** | three `weights/*.pth` | **E196–E200** only; 4 final-stage derived PNGs | [README](model6_rfdetr_n_hybrid/README.md); `ARCHIVE_MANIFEST.json` |
| Model 7 | 07 | D-FINE Nano | Intended synthetic-only | **Not archived** | — | — |
| Model 8 | 08 | D-FINE Nano | Intended hybrid | **Not archived** | — | — |

**Interpretation:** an E200 completion receipt means the training run reached its target; it does **not** automatically mean every epoch's metric logs, every best checkpoint, or every GPU-worker-produced diagnostic image is present on GitHub.

## Dataset and label contract

The six class names, with canonical YOLO IDs 0–5:

```text
0 leaf_algal
1 leaf_blight
2 leaf_colletotrichum
3 leaf_healthy
4 leaf_phomopsis
5 leaf_rhizoctonia
```

For the RF-DETR COCO-format train/validation dataset, categories are numbered **1–6** and must be mapped to the canonical **0–5** IDs for cross-architecture evaluation.

- **Synthetic-only run lineage:** V4.1, Models 1/3/5.
- **Updated hybrid run lineage:** corrected V4.2, Models 2/4/6.
- V4.2 train/val: **1,425 / 399 images**.
- V4.2 dataset archive SHA-256: `19923e8d08acab1b66de2c0e3b94a56abaabf7051e8d462e929070daf3958834`.
- Model 6's COCO conversion logged **35,701 train boxes / 11,879 val boxes**.

**Study limitation:** because synthetic and hybrid conditions are published from different dataset versions, apparent performance differences cannot be attributed solely to adding hybrid data. Training schedules/hyperparameters should also be checked per run before claiming a controlled pairwise comparison.

## Architecture and artifact conventions

### YOLO26n: Models 1 and 2

- Inference weights: `best.pt`.
- Run02 includes `weights/last.pt`, several saved intermediate checkpoints, `results/results.csv` and **17 original image artifacts** (PNG/JPG).
- Run01 has `results/results.csv`, derived graphs and original diagnostics; the CSV stops at E196 despite a separate E200 completion receipt.

### YOLO26n-P2: Models 3 and 4

- Architecture adds the P2 small-object detection scale.
- Run03 contains the complete 200-epoch results CSV and derived training-history figures.
- Run04 includes **E1–E200** merged metrics, a separate **E156–E200** stage CSV, and **17 original training PNG/JPG artifacts**.
- Run04's **recorded all-history best** is **E167 / validation mAP50–95 = 0.56938**, with `best.pt` and E200 `last.pt` traced in its manifest.
- Run04's actual effective settings include `nbs=16` and `patience=0`; do not assume all similarly named runs used identical effective settings.
- Run04's stored `results.png` came from the final resumed worker stage; it must not be presented as necessarily plotting the entire merged E1–E200 CSV.

### RF-DETR Nano: Models 5 and 6

- Uses **384-pixel** architecture-native resolution, **batch 4**, **gradient accumulation 4**, **200 epochs**, **AdamW**, **EMA** and the recorded LR schedule dropping at epoch 160.
- Model 5 uses V4.1 Synthetic-only and publishes three `.pth` inference weights plus a full E200 `last.ckpt`.
- Model 6 uses V4.2 corrected Hybrid and publishes `checkpoint_best_ema.pth`, `checkpoint_best_total.pth`, and `last_ema.pth`.
- **Model 6 does not currently publish a full E200 Lightning `.ckpt` on GitHub**; rolling full-resume `.ckpt` files are held in Kent Master Drive.
- Model 6 `results/metrics_E196_E200.csv` and its four `results/png/*_E196_E200.png` files cover **only the last five epochs**. They must not be described as full-history curves or official prediction-level PR/F1-confidence diagrams.
- Model 6's published “best” EMA/total files originate from the **final Johny Runtime**, not an independently validated *global* best over all worker sessions. See [Model 6's limitations](model6_rfdetr_n_hybrid/README.md).

## Training, metrics, and evaluation integrity

1. **Validation vs test:** archived metrics are validation metrics, **not sealed-test performance**.
2. **Global best:** to claim global best for a multi-Colab handoff, verify all session histories and candidate best weights, rather than just taking the last session's file.
3. **Missing epochs:** do not fill absent rows with interpolated or fabricated values. Model01 reports E1–E196; Model05 E151–E200; Model06 E196–E200.
4. **Derived vs original images:** PNGs regenerated from aggregated epoch statistics are distinct from Ultralytics confidence sweeps, confusion matrices and real prediction examples.
5. **Historical reports:** [`METRICS_REPORT.md`](METRICS_REPORT.md) is a **V4.1 study report**, so its old Model02/04 hybrid values **must not** be attributed to their current V4.2 versions. [`V4_2_HYBRID_RETRAINING_PLAN.md`](V4_2_HYBRID_RETRAINING_PLAN.md) is a **pre-execution plan** that still says training had not started.
6. **Legacy RF-DETR verification:** [`RFDETR_RUN05_RUN06_GITHUB_LFS_VERIFICATION.json`](RFDETR_RUN05_RUN06_GITHUB_LFS_VERIFICATION.json) documents the older **V4.1** Run06 and **does not match** today's Model06 V4.2 final weights; use the new [Model 6 manifest](model6_rfdetr_n_hybrid/ARCHIVE_MANIFEST.json).

## Retrieving actual weights

```bash
git clone --branch model-synthetic-and-hybrid https://github.com/JohnyChia/durian_training.git
cd durian_training
git lfs install
git lfs pull
```

GitHub's contents API often reports the **small Git LFS pointer** (about 132–134 bytes), not the actual multi-megabyte weight. Verify full payloads against the SHA-256 in each model-specific manifest. RF-DETR `.pth` weights are **not equivalent** to YOLO `.pt` weights; Lightning `.ckpt` also contains full resume state.

## Useful source documents

- [Root README](../README.md) — overview and caveats.
- [Run02 E200 manifest](model2_yolo26n_hybrid/MODEL2_E200_MANIFEST.json).
- [Run04 archive manifest](model4_yolo26n_p2_hybrid/ARCHIVE_MANIFEST.json) and [Run04 final audit](model4_yolo26n_p2_hybrid/FINAL_AUDIT.json).
- [Run05 artifact status](model5_rfdetr_n_synthetic/results/ARTIFACT_STATUS.md) and [training configuration](model5_rfdetr_n_synthetic/results/training_config.json).
- [Run06 archive manifest](model6_rfdetr_n_hybrid/ARCHIVE_MANIFEST.json) and [E200 receipt](model6_rfdetr_n_hybrid/results/JOHNY_E200_AUDIT.json).
- [Separate controlled-experiment protocol](../experiment/controlled_experiment/README.md): its **100-epoch** planned jobs are not the **200-epoch** archived runs described above.

## Outstanding archival tasks

- Reconstruct all actual Model 6 metric segments and validate the globally best EMA checkpoint across Johny / Student / Kent handoffs.
- Reconcile run-specific missing metrics and archived images without falsifying provenance.
- Audit current V4.2 Hybrid against V4.1 Synthetic under the same evaluation pipeline; avoid reusing old V4.1 Hybrid leaderboard entries.
- Train and archive the remaining D-FINE Nano runs only when their provenance and artifacts have been verified.
