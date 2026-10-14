# Current 8-run Study Model Artifacts

This branch temporarily contains only the completed V4.1 Synthetic-only
models. Historical V4.1 Hybrid artifacts were removed before V4.2 retraining
and remain recoverable from tag `archive/v4.1-hybrid-before-v42`.

| Run | Architecture | Condition | Artifact |
| --- | --- | --- | --- |
| 01 | YOLO26n | Synthetic-only | `model1_yolo26n_synthetic/best.pt` |
| 03 | YOLO26n-P2 | Synthetic-only | `model3_yolo26n_p2_synthetic/best.pt` |
| 05 | RF-DETR Nano | Synthetic-only | `model5_rfdetr_n_synthetic/weights/checkpoint_best_ema.pth` |

## Integrity

### Run 01

`b7e221be1716737941f9ac0bfeab832e6fce67f405a1d0f133287ce84e2b4ef0`

### Run 03

SHA256:

`a8ae50bfea646f34849961be3204b9860dcf7c7e87de159498c7b59872fd369d`

### Run 05 (best EMA)

`f2912b418819bc6732958f103b07708c95e3bad15219e5cbcc9f0bccda7ef744`

New Run02, Run04 and Run06 directories must only be created from completed
V4.2 Hybrid E200 training artifacts initialized from the locked pretrained
initializers. Old V4.1 Hybrid checkpoints or results must not be relabelled as
V4.2 outputs.
