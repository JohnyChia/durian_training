# Run05 — RF-DETR Nano Synthetic, V4.1 (E200)

Source: Kent Google Drive archived Run05 training bundle, `MODEL5_MANIFEST.json`.

## Weights / checkpoints (actual bytes stored on Drive)

| Artifact | SHA-256 | Google Drive ID | GitHub LFS verified? |
|---|---|---|---|
| checkpoint_best_ema.pth | f2912b418819bc6732958f103b07708c95e3bad15219e5cbcc9f0bccda7ef744 | 1W84mKXeyfX7v0x96cV77cHglpReQCQoR | NO — not uploaded to Run05 path |
| checkpoint_best_total.pth | 4345a1f7bb8af91a2d215d63d69f52731f2bc411818e0686970bee1770a0d0a4 | 1CGYHquBG-VIbtDJH1rVmuuWoLg1Bj7q3 | NO |
| last_ema.pth | f3210262b886fd531aa6e4d22eb0201ced9f8eb6b4b1fac3fa919750e13e7c3c | 1xqDqQ6MilEdZCMxEmuIU943l8jrN_aM1 | NO |

Results in this repository: E200 manifest, handoff receipt, verified inputs, training configuration, metrics CSV, hparams, and charts reconstructed from stored validation rows.

**Plots:** `images/validation_metrics.svg`, `images/classification_metrics.svg`; source `metrics.csv`, E151–E200 **only**. They are not complete 200-epoch curves or post-hoc V4.2 evaluation.

### Science / evaluation note

Model trained on V4.1 synthetic-only. Evaluation cohorts must be explicitly stated; do not represent V4.1 validation numbers as V4.2 independent cohort performance. SHA-256 must be recomputed from actual downloaded bytes for intake.
