# Model 6 / Run06 — RF-DETR Nano, Corrected V4.2 Hybrid

**Status: E200 training completed; GitHub artifacts uploaded.**  
GitHub branch: `model-synthetic-and-hybrid`  
Directory: `models_8run/model6_rfdetr_n_hybrid/`  
Archive commit: [`1a2be4198e046fa559a3037363f417021281c0cc`](https://github.com/JohnyChia/durian_training/commit/1a2be4198e046fa559a3037363f417021281c0cc)

> **Important limitation:** Training reached E200, but this folder only contains the **final E196–E200 metric segment**. Its saved `checkpoint_best_ema.pth` is the **last Colab runtime's best**, **not an independently demonstrated all-E1–E200 best**. Full historical metrics and globally best EMA selection remain pending.

## Purpose and dataset provenance

Model 6 is the RF-DETR Nano **Hybrid** partner to [Model 5 (V4.1 Synthetic-only)](../model5_rfdetr_n_synthetic/). It uses the **corrected V4.2 Hybrid train/validation data**, **not V4.1 Hybrid** and **not the V4.1 synthetic-only dataset**.

| Input | Verified value |
| --- | --- |
| Dataset condition | V4.2 corrected Hybrid |
| Original input | `v42_hybrid_trainval.tar.gz`, Kent Master Drive |
| Archive size | 334,759,590 bytes |
| Archive SHA-256 | `19923e8d08acab1b66de2c0e3b94a56abaabf7051e8d462e929070daf3958834` |
| Train images / boxes | **1,425 / 35,701** |
| Validation images / boxes | **399 / 11,879** |
| Number of classes | 6 |
| Training annotation source | Canonical YOLO labels, converted to COCO for RF-DETR |
| COCO category mapping | IDs 1–6 correspond to canonical IDs 0–5 |

Canonical class order: `leaf_algal`, `leaf_blight`, `leaf_colletotrichum`, `leaf_healthy`, `leaf_phomopsis`, `leaf_rhizoctonia`.

No sealed-test result is asserted by this archive.

## Architecture, initializer, and training recipe

| Setting | Value |
| --- | --- |
| Architecture | RF-DETR Nano (RFDETRNano; LWDETR implementation) |
| Model parameters shown by Lightning | About 30.2 million trainable |
| Pretrained initialization | Official `rf-detr-nano.pth`; **90-class head replaced with 6-class head** |
| Initializer SHA-256 | `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d` |
| Output classes | 6 |
| Resolution | 384 |
| Epoch target/completion | 200 / 200 |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch 16) |
| Optimizer | AdamW |
| LR / encoder LR | 0.0001 / 0.00015 |
| Weight decay | 0.0001 |
| EMA decay / tau | 0.993 / 100 |
| Learning-rate schedule | Step; `lr_drop=160` |
| Early stopping | Disabled |
| Validation interval | Each epoch |
| Checkpoint interval | Each 5 epochs through rotating cloud slots |
| Seed | 20260917 |
| Verified run software baseline | RF-DETR 1.11.0; PyTorch Lightning 2.6.6 |

The experiment deliberately used **the same verified RF-DETR Nano initializer SHA-256 and core training settings as Model 5**, while changing the dataset to V4.2 Hybrid. Because Model 5 uses V4.1 data, this is *not* a pure single-variable controlled comparison of synthetic versus hybrid data.

### Multi-Colab training lineage

The training was continued between **Johny**, **Student** and **Kent** Colab GPU workers, with a shared **Kent Master Drive** as checkpoint storage. Complete Lightning `.ckpt` files were used for recovery with model parameters, optimizer states, LR scheduler state and global step. Runs used five-epoch A/B rotation. Workers later used account-specific slot names because attempts to delete a different account's old Drive objects could return 403 permission errors.

The published [E200 completion audit](results/JOHNY_E200_AUDIT.json) records:

- Last worker: Johny Colab.
- Source of final resume: `kent_slot_A.ckpt`.
- Source resume `global_step`: **17550** (this is the *incoming* checkpoint's step, **not** the E200 final step).
- Final training completion: **E200**.
- Five final-runtime files with sizes and SHA-256 checksums.

The cloud slot lineage and audit support completed resumed training, but are **not** a substitute for a consolidated E1–E200 metrics receipt or an all-worker global-best assessment.

## GitHub release contents

```text
model6_rfdetr_n_hybrid/
├── README.md
├── ARCHIVE_MANIFEST.json
├── weights/
│   ├── checkpoint_best_ema.pth
│   ├── checkpoint_best_total.pth
│   └── last_ema.pth
└── results/
    ├── JOHNY_E200_AUDIT.json
    ├── hparams.yaml
    ├── metrics_E196_E200.csv
    └── png/
        ├── ema_map_E196_E200.png
        ├── f1_E196_E200.png
        ├── loss_E196_E200.png
        └── precision_recall_E196_E200.png
```

### Weight meaning and integrity

| Weight | Purpose / scope | Actual file size | SHA-256 |
| --- | --- | ---: | --- |
| [`checkpoint_best_ema.pth`](weights/checkpoint_best_ema.pth) | Last worker stage's best EMA inference weight | 120,974,371 bytes | `b333edab9b69d7c85c6405fa468ad767c820adbe12825bcdf661dc5aa069c266` |
| [`checkpoint_best_total.pth`](weights/checkpoint_best_total.pth) | Last worker stage's best total/non-EMA weight | 120,962,691 bytes | `c9e05e3f894e54ca9c4a1b565d52e811aa64a13bc2edd23babab78d746ac4478` |
| [`last_ema.pth`](weights/last_ema.pth) | Last EMA inference weight at training end | 120,969,071 bytes | `56180dfd84b22695ee1363403c4c08e483a2836089319791332191ee5611ada5` |

These are **RF-DETR `.pth`** weights. They are **not YOLO `best.pt` files**, and simply renaming them to `.pt` would not make them compatible with Ultralytics YOLO loading. All three binaries were reported pushed successfully via **Git LFS** (3/3; ~363 MB); the branch HEAD and Git LFS pointers are present on GitHub. Use the [archive manifest](ARCHIVE_MANIFEST.json) for file hashes and full byte sizes.

**No full E200 Lightning `.ckpt` has been published under this model directory.** The worker-to-worker full resume checkpoints are maintained separately on Kent Master Drive. Inference weights do not include the same optimizer/scheduler resume guarantees.

### Results and figures

- [`metrics_E196_E200.csv`](results/metrics_E196_E200.csv): five rows for completed epochs **196–200**. Original RF-DETR CSV epoch values are **zero-based (195–199)**.
- [`ema_map_E196_E200.png`](results/png/ema_map_E196_E200.png): derived final-stage EMA mAP trend.
- [`f1_E196_E200.png`](results/png/f1_E196_E200.png): derived final-stage F1 trend from the recorded metrics.
- [`loss_E196_E200.png`](results/png/loss_E196_E200.png): derived final-stage loss trends from recorded fields.
- [`precision_recall_E196_E200.png`](results/png/precision_recall_E196_E200.png): **per-epoch aggregate precision/recall**, not a prediction-level confidence-threshold PR curve.
- [`JOHNY_E200_AUDIT.json`](results/JOHNY_E200_AUDIT.json): source worker's E200 receipt and recorded source file SHA-256.

The four PNGs were **regenerated from E196–E200's recorded CSV** because the final Drive runtime archive did not contain original PNGs. They are faithfully **labelled final-segment derived plots**, **not** E1–E200 charts, original confusion matrices, or original RF-DETR trainer figures.

## What is verified — and what is not?

| Claim | Status |
| --- | --- |
| Model 6 reached E200 according to completion audit | **Verified from receipt** |
| V4.2 input archive SHA-256 | **Verified** |
| Model 5-matching initializer SHA-256 | **Verified** |
| Three final-runtime `.pth` files and source SHA-256 manifest | **Verified in the archive workflow** |
| GitHub commit and three Git LFS pointers | **Verified** |
| E196–E200 metric CSV and four derived PNGs | **Available** |
| Original contiguous E1–E200 metrics | **Not yet reconstructed** |
| **Global** highest-scoring EMA across all worker sessions | **Not yet verified** |
| Original whole-run trainer plots / confusion matrices | **Not available in this archive** |
| Sealed-test or real-world detector performance | **Not evaluated here** |

## Reproducibility and follow-up

Clone the project with Git LFS enabled before attempting to use the weights:

```bash
git clone --branch model-synthetic-and-hybrid https://github.com/JohnyChia/durian_training.git
cd durian_training
git lfs install
git lfs pull
```

To complete a **global-best audit**, recover all session metric CSVs and candidate EMA weights from the Johny/Student/Kent checkpoints and final archives; align zero-indexed epoch values to completed E1–E200 epochs; compare the recorded validation selection metric; verify file hashes; and publish an explicit consolidated manifest. Until then, **do not** describe `checkpoint_best_ema.pth` as the globally best Run06 model.

For the complete 8-run inventory and version caveats, see [`../README.md`](../README.md). For Model 5's synthetic-only counterpart, see [`../model5_rfdetr_n_synthetic/`](../model5_rfdetr_n_synthetic/).
