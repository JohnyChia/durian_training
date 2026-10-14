# V4.2 Hybrid Retraining Plan

Status: **PLANNED — TRAINING NOT STARTED**

The historical V4.1 Hybrid artifacts are recoverable from Git tag
`archive/v4.1-hybrid-before-v42`. They must not be resumed, copied, or
represented as V4.2 results.

## Frozen dataset input

- Lineage: `V4_2_HYBRID_SOURCE_FAMILY_CORRECTED`
- Train images: 1,425
- Validation images: 399
- Source-family correction manifest SHA-256:
  `b634760affc11ce33f4dcbda5904aa4c482480824a3650b2d70b9add45a062bd`
- Colab archive SHA-256:
  `19923e8d08acab1b66de2c0e3b94a56abaabf7051e8d462e929070daf3958834`

## Future runs

| Run | Future directory | Architecture | Required fresh initializer |
|---|---|---|---|
| Run02 | `model2_yolo26n_hybrid/` | YOLO26n | `yolo26n.pt`, SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| Run04 | `model4_yolo26n_p2_hybrid/` | YOLO26n-P2 | `yolo26n_p2_semantic_init.pt`, SHA-256 `fa444a8aebf53262031d25a4b45e83270b4625ccf0a9c883479bc03917b6eaa2` |
| Run06 | `model6_rfdetr_n_hybrid/` | RF-DETR Nano | `rf-detr-nano.pth`, SHA-256 `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d` |

Each job must start from the stated initializer. Resuming from a V4.1 E200
checkpoint is prohibited.

## Required completion evidence

Do not create the future model directory on this branch until its training job
has completed successfully. Each new directory must contain or reference:

1. the frozen effective training configuration, seed, package versions, GPU
   environment, and V4.2 dataset/archive identities;
2. the complete E200 checkpoint and optimizer/scheduler state required by the
   formal protocol;
3. the selected best-validation checkpoint and selection metric receipt;
4. loss and validation metric histories plus generated plots;
5. SHA-256, byte size, class order, threshold provenance, architecture, and
   initializer lineage for every published checkpoint;
6. a manifest explicitly stating `V4_2_HYBRID_SOURCE_FAMILY_CORRECTED`;
7. a Google Drive handoff receipt for Kent master checkpoint storage.

No Synthetic Dev, Real Dev, AI-assisted Real reference, sealed-test data, or
evaluation output may enter the training archive or influence checkpoint
selection. Sealed-test access remains prohibited until the final release gate.
