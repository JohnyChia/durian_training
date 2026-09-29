# Durian leaf model experiment tooling

This directory preserves/audits the canonical dataset and supplies the guarded,
framework-neutral control plane for the four-candidate detector study. No model
has been trained by this preparation.

## Retained tools

- `audit_clean_dataset.py` independently validates integrity, class mapping, distributions, and split leakage.

Run the safe dataset audit with:

```bash
python3 experiment/training_model/audit_clean_dataset.py
```

The accepted result is:

```text
PASS — GAZEBO TRAINING READY / REAL ANNOTATION PENDING
```

## Controlled training status

The retired Original, Baseline, and EfficientNet Advanced entry points were removed after dependency analysis. They were not four controlled architecture candidates, and their shared runner automatically exposed the test split after each run.

The executable frozen configurations now cover:

1. YOLO26n
2. YOLO26n-P2
3. RF-DETR-N
4. D-FINE-N

Run the single behavioral gate with:

```bash
python3 experiment/training_model/readiness_audit.py
```

It re-hashes canonical data, reconstructs identities, verifies every YOLO/COCO
box, validates config/code/checkpoint locks, exercises the common evaluator and
adapters, proves unauthorized test evaluation is denied, and checks the GPU
preflight report. The accepted final line is `PASS — CONTROLLED TRAINING READY`.

Use `candidate_runner.py --check` to validate an individual configuration.
`--execute` is the deliberate future training action; do not use it until the
isolated dependency environments and GPU batch preflight have been reviewed.
Training and selection receive only train/validation. The RF-DETR trainer view
physically omits test. Test evaluation requires a release manifest created by
`test_guard.py freeze` from final frozen checkpoints.

See `experiment/controlled_experiment/README.md` for the experiment contract
and exact next commands. The retired `experiment/yaml/` pipeline and one-time
dataset rebuild utility are intentionally excluded from the GitHub package.

## Environment note

`requirements.txt` remains limited to dataset tooling. Framework environments
are intentionally separate and frozen under `controlled_experiment/locks/`.
The current virtual environment is not the training environment; no dependency
was installed during preparation.
