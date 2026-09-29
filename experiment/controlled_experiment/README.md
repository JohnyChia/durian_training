# Controlled four-detector experiment

Status: configurations and controls are frozen; training has not started.

## Contract

- Candidates: YOLO26n, YOLO26n-P2, RF-DETR Nano, D-FINE Nano.
- Canonical source: `experiment/datasets`; it is never written by this tooling.
- Classes: canonical IDs 0–5 in the order recorded in every config.
- Development: train fits; validation selects checkpoints/configs/thresholds.
- Test: sealed until a release manifest binds final configs, dataset view, and
  final checkpoint hashes. Post-test changes constitute a new study.
- Reporting: all predictions are normalized to
  `image_id,class_id,confidence,x1,y1,x2,y2` and scored by the same evaluator.

The COCO view stores metadata and symlinks only. Its 1-based category IDs map
explicitly to canonical 0-based IDs. Global deterministic image IDs are stable
across split files. RF-DETR sees only `train/` and `valid/`; no test folder is
present in its trainer view.

## Frozen recipes

| Candidate | Initialization | Input | Budget / batch | Optimizer and LR | Early stop |
|---|---|---:|---|---|---|
| YOLO26n | official YOLO26n COCO | 640 | fixed 100 / 4, nominal 16 | AdamW, 1e-3, cosine | disabled (`patience=0`) |
| YOLO26n-P2 | semantic YOLO26n transfer; new P2 random | 640 | fixed 100 / 4, nominal 16 | AdamW, 1e-3, cosine | disabled (`patience=0`) |
| RF-DETR-N | official Nano COCO | native 384 | fixed 100 / 4×4 accumulation | AdamW, 1e-4; step at epoch 80 | disabled (`early_stopping=False`) |
| D-FINE-N | official Nano COCO | native 640 | fixed 100 / 4 | AdamW, 2.5e-5 scaled | disabled; fixed budget |

Every candidate uses seeds 20260917, 20260923, and 20261001, for 12 controlled
runs. All receive exactly 100 epochs; validation may select the best checkpoint,
but cannot shorten a run. The architecture-native RF-DETR resolution is a
declared difference, not an unnoticed default. RF-DETR's epoch-80 LR drop
preserves the prior recipe's 160/200 (80%) schedule position using the pinned
1.11.0 `lr_scheduler_kwargs` API.

D-FINE preserves the upstream Nano recipe's 12-epoch final refinement stage:
augmentation and multiscale collation stop at epoch 88 instead of 148. This is
separate from its upstream `MultiStepLR` milestone 500, which remains unchanged
and dormant, as it was in the 160-epoch recipe. Linear warmup remains 500
optimizer steps. `remap_mscoco_category: true` maps the generated COCO category
IDs 1–6 to D-FINE labels 0–5 during training and maps predictions back for COCO
evaluation.

The trainer inputs expose only train and validation. In particular, the runtime
YOLO YAML removes the canonical YAML's `test:` entry, RF-DETR receives a view
containing only `train/` and `valid/`, and D-FINE receives only train/val paths.
The sealed test split is unavailable to training, tuning, early stopping,
checkpoint selection, and sanity runs.

No official YOLO26-P2 weights exist. `semantic_transfer_yolo26_p2.py` maps by
graph role and then shape: backbone and P5→P3 neck are identical; later P4/P5
PAN nodes shift by six; Detect P3/P4/P5 branches shift from 0/1/2 to 1/2/3.
New P2 nodes and Detect branch 0 remain seeded random. The verification copied
708 tensors (2,591,962 elements; 96.56% of target state) and passed a 640 forward.

## Safe commands now

```bash
python3 experiment/training_model/readiness_audit.py
python3 experiment/training_model/verify_coco_view.py
python3 experiment/training_model/common_evaluator.py --self-test
python3 experiment/training_model/gpu_preflight.py \
  --output experiment/controlled_experiment/reports/gpu_preflight.json
```

After dependency/GPU review, place the three hash-verified official checkpoints
in `controlled_experiment/weights/`, generate the P2 initialization, then use
`candidate_runner.py --execute --seed <frozen-seed>`. This is intentionally an
explicit action and is not part of readiness auditing.

After all validation decisions are final, freeze final checkpoints:

```bash
python3 experiment/training_model/test_guard.py freeze \
  --checkpoint /absolute/path/to/final-checkpoint \
  --output experiment/controlled_experiment/releases/final.json
```

Pass that release file to `common_evaluator.py --release ...` for the single
final test evaluation. A hash-bound `.used.json` receipt is written after a
successful test evaluation; the same release cannot be reused. Never use test
results to alter the frozen study.
