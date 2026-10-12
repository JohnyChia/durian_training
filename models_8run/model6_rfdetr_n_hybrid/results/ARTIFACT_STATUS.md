# Run06 — RF-DETR Nano Hybrid, V4.1 (E200)

**Training dataset:** `/content/durian_v4_1_8run_final/datasets/coco/hybrid`, **1415 train / 406 validation**.

## Stored artifacts

| Artifact | SHA-256 | Google Drive file ID | GitHub status |
|---|---|---|---|
| best EMA weight (.pth) | fb2bb0775704c5ccc4bbe01b8dbe776edf23ce5e6a3073ef797a120668ce2203 | 12QCjfbUAS9-iqqrx8pHWEgfzY1Ijp4LE | LFS pointer exists; full LFS object NOT verified |
| best Total weight (.pth) | 0a4bd4e420a28a1e39ffb68ef006defe80228372f4796a9b846dd6e5cc61a1d0 | 1XSYq5hMCwesVzjTEUo-qTLhr-W7cGzc5 | LFS pointer exists; full LFS object NOT verified |
| E200 full-state ckpt (.ckpt) | 203dff45b9906d506461d2c060e2feb4fa603c67e326ec95b0308f1591fe21ba | 1h6fRE8mcOl_zcRcZKWk7opRUC-8I6drD | Drive only |

Results archived: `metrics.csv`, `training_config.json`, `ckpt_e200_manifest.json`, plus SVG charts built from real validation rows.

**Plots:** `images/validation_metrics.svg` and `images/classification_metrics.svg` represent only **E191–E200**, as contained in stored `metrics.csv`. No source-authenticated original PNG plots or original PR/confusion curves were available; do not interpret these as full 200-epoch history.

**Lineage:** `V4_1_HYBRID_EXPOSURE_BIASED`. V4.1 family leakage and historical checkpoint selection bias are documented. A new V4.2 cohort evaluation does not erase past training/checkpoint selection exposure. This checkpoint must NOT be mislabeled V4.2-trained or leakage-free.

For final model intake, obtain actual files and compute SHA-256, rather than trusting an LFS pointer alone.
