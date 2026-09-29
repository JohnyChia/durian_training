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
| YOLO26n | official YOLO26n COCO | 640 | 200 / 4, nominal 16 | AdamW, 1e-3, cosine | patience 40 |
| YOLO26n-P2 | semantic YOLO26n transfer; new P2 random | 640 | same as YOLO26n | same as YOLO26n | patience 40 |
| RF-DETR-N | official Nano COCO | native 384 | 200 / 4×4 accumulation | AdamW, 1e-4 | patience 40 |
| D-FINE-N | official Nano COCO | native 640 | official 160 / 4 | AdamW, 2.5e-5 scaled | disabled; fixed official schedule |

Every candidate uses seeds 20260917, 20260923, and 20261001. The architecture-
native RF-DETR resolution is a declared difference, not an unnoticed default.
D-FINE uses its fixed upstream schedule and best-validation selection because
its pinned trainer has no native early-stopping contract.

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
