# Run06 — GitHub LFS verified E200 artifacts

Training lineage: `V4_1_HYBRID_EXPOSURE_BIASED`.
Seven Run05/Run06 source objects were downloaded from Kent Drive,
SHA256-verified, uploaded to GitHub LFS, independently fetched
from a fresh GitHub clone, and SHA256-verified again.

| Artifact | Size in bytes | SHA256 |
|---|---:|---|
| `models_8run/model6_rfdetr_n_hybrid/weights/checkpoint_best_ema.pth` | 120974499 | `fb2bb0775704c5ccc4bbe01b8dbe776edf23ce5e6a3073ef797a120668ce2203` |
| `models_8run/model6_rfdetr_n_hybrid/weights/checkpoint_best_total.pth` | 120962819 | `0a4bd4e420a28a1e39ffb68ef006defe80228372f4796a9b846dd6e5cc61a1d0` |
| `models_8run/model6_rfdetr_n_hybrid/checkpoints/last.ckpt` | 483442431 | `203dff45b9906d506461d2c060e2feb4fa603c67e326ec95b0308f1591fe21ba` |

See `models_8run/RFDETR_RUN05_RUN06_GITHUB_LFS_VERIFICATION.json`.

**Limitation:** V4.1 hybrid source-family leakage / historical
checkpoint-selection bias remain. This was NOT trained on V4.2.
