# Durian Leaf Disease Detection — Training and Model Archive

This repository records durian leaf-disease object-detection experiments, model checkpoints, training logs, evaluation tooling, and reproducibility evidence. This README describes the **`model-synthetic-and-hybrid`** branch, especially its **eight-run model archive** in [`models_8run/`](models_8run/README.md).

> **Current archive status (18 October 2026):** Models **1–6** have E200 completion evidence and published model artifacts. **Models 7–8 are not yet archived** on this branch. Training completion and archival completeness are different: not every completed run has a full E1–E200 metric history or a globally verified best checkpoint.

## Study layout

The planned eight-run matrix has four detector families, each with a synthetic-only and a hybrid training condition:

| Detector | Synthetic-only (V4.1) | Hybrid (corrected V4.2) |
| --- | --- | --- |
| YOLO26n | [Model 1 / Run01](models_8run/model1_yolo26n_synthetic/) — archived | [Model 2 / Run02](models_8run/model2_yolo26n_hybrid/) — archived |
| YOLO26n-P2 | [Model 3 / Run03](models_8run/model3_yolo26n_p2_synthetic/) — archived | [Model 4 / Run04](models_8run/model4_yolo26n_p2_hybrid/) — archived |
| RF-DETR Nano | [Model 5 / Run05](models_8run/model5_rfdetr_n_synthetic/) — archived | [Model 6 / Run06](models_8run/model6_rfdetr_n_hybrid/) — archived |
| D-FINE Nano | Model 7 — not archived | Model 8 — not archived |

**Dataset-version caveat:** the synthetic-only runs use **V4.1**, while the current hybrid runs use **corrected V4.2**. Comparisons are therefore not isolated tests of "synthetic versus hybrid" alone: dataset versions and potentially effective training settings differ. No sealed-test performance is established by the training files in this archive.

The six canonical leaf-disease classes (zero-based training IDs) are:

```text
0 leaf_algal
1 leaf_blight
2 leaf_colletotrichum
3 leaf_healthy
4 leaf_phomopsis
5 leaf_rhizoctonia
```

For RF-DETR's COCO training view, category IDs **1–6** correspond to these canonical labels **0–5**. Keep the mapping explicit when evaluating or joining predictions.

## What is actually published?

| Run | Main model artifacts | Available recorded history | Training completion |
| --- | --- | --- | --- |
| 01 | `best.pt` | E1–E196 in `results/results.csv` | E200 reported; final four CSV rows unavailable |
| 02 | `best.pt`; `weights/last.pt`; E180–E195 checkpoints | `results/results.csv` and 17 original training images | E200 |
| 03 | `best.pt` | E1–E200 in `results/results.csv` | E200 |
| 04 | `best.pt`, `last.pt` | E1–E200 merged `results.csv`, stage CSV and 17 images | E200 |
| 05 | best/final RF-DETR `.pth` and `checkpoints/last.ckpt` | E151–E200 in `results/metrics.csv` | E200 |
| 06 | three best/final RF-DETR `.pth` weights | **E196–E200 only** in `results/metrics_E196_E200.csv` | E200 |

Detailed paths, integrity manifests, provenance limitations and figures are documented in [the model archive README](models_8run/README.md). **Do not infer absent metric rows or claim that a final-runtime "best" is the global E1–E200 best without the historical evidence.**

## Data lineage and experimental boundaries

- **V4.1 Synthetic-only:** Models 1, 3 and 5. Do not relabel these data as V4.2.
- **Corrected V4.2 Hybrid:** Models 2, 4 and 6. Training view: **1,425 train / 399 validation images**, across six classes.
- V4.2 training archive SHA-256: `19923e8d08acab1b66de2c0e3b94a56abaabf7051e8d462e929070daf3958834`.
- The study uses validation data for model selection. **Validation metrics are not sealed-test, real-world, or deployment performance.**
- Do not contaminate training or checkpoint selection with sealed-test images/results.

The [controlled experiment package](experiment/controlled_experiment/README.md) is **a separate frozen protocol**, with its own run budget and seed plan. Its `100 epochs` specification is **not** the training recipe for the separately completed 200-epoch artifacts under `models_8run/`.

## Working with model artifacts

Large weights are stored using **Git LFS**, not ordinary Git objects. After cloning this branch, install and fetch LFS objects:

```bash
git clone --branch model-synthetic-and-hybrid https://github.com/JohnyChia/durian_training.git
cd durian_training
git lfs install
git lfs pull
```

A `.pt`, `.pth` or `.ckpt` file shown as approximately 130 bytes in GitHub's contents API is generally the **Git LFS pointer**, not the actual model. Confirm the full binary size and SHA-256 against each run's manifest before inference or comparison.

- **YOLO26n / YOLO26n-P2:** `.pt` weight format.
- **RF-DETR Nano:** `.pth` model/inference weights; a full Lightning `.ckpt` additionally contains optimizer/scheduler and resume state. **Renaming `.pth` to `best.pt` does not convert formats.**
- **Shared Kent Master Drive:** preserves rotating worker-to-worker resume checkpoints. Colab GPU execution moved between Johny, Student and Kent accounts. Cloud checkpoints should not be deleted until global-best and history recovery is complete.

## Reports and provenance

| Document | Intended use / important limitation |
| --- | --- |
| [`models_8run/README.md`](models_8run/README.md) | **Current** model 1–6 archive directory, status and limitations |
| [Model 4 README](models_8run/model4_yolo26n_p2_hybrid/README.md) | V4.2 Hybrid YOLO26n-P2 results; recorded best E167, mAP50–95 0.56938 |
| [Model 6 README](models_8run/model6_rfdetr_n_hybrid/README.md) | V4.2 Hybrid RF-DETR Nano E200 handoff and final-stage artifacts |
| [`models_8run/METRICS_REPORT.md`](models_8run/METRICS_REPORT.md) | **Historical V4.1 study**; its Model 2/4 figures must **not** be treated as current V4.2 results |
| [`models_8run/V4_2_HYBRID_RETRAINING_PLAN.md`](models_8run/V4_2_HYBRID_RETRAINING_PLAN.md) | Historical pre-retraining plan; its "not started" status is superseded by completed Run02/04/06 evidence |
| [`models_8run/RFDETR_RUN05_RUN06_GITHUB_LFS_VERIFICATION.json`](models_8run/RFDETR_RUN05_RUN06_GITHUB_LFS_VERIFICATION.json) | Historical verification referring to the **older V4.1** Run06 lineage; it does **not** authenticate the newly archived **V4.2 Model 6** weights |
| [`experiment/controlled_experiment/README.md`](experiment/controlled_experiment/README.md) | Separate controlled-evaluation contract, not proof that all eight archival runs were executed under that contract |

## Remaining work

1. Recover and validate Model 6's complete E1–E200 metric history and verify the globally best EMA across all workers.
2. Reconcile missing metric rows and training-time artifacts where the existing archive is incomplete; preserve original sources and mark any generated figures as derived.
3. Complete and archive Models 7 and 8 only after their training and provenance pass audit.
4. Run a common, leakage-controlled evaluator and publish a separately identified final comparison; do not merge training-validation scores with sealed-test findings.

No missing scores, missing weights, or unperformed evaluations are implied by this README.
