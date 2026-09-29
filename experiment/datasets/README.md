# Canonical Durian Leaf Detection Dataset

This is the single canonical detector dataset. It was rebuilt from surviving repository evidence with seed `20260923`.

## Scope

**GAZEBO TRAINING/EVALUATION DATA AVAILABLE. REAL ANNOTATION PENDING.**

Official `images/` and `labels/` contain only reviewed Gazebo samples. No legacy Real collage-tile labels and no legacy 0.98 full-image labels are official ground truth. Surviving Real photographs are preserved under `domains/real/unlabelled_source/`; legacy collages are preserved without labels under `auxiliary/legacy_collage/` because the original source photographs are absent.

## Classes

0. `leaf_algal`
1. `leaf_blight`
2. `leaf_colletotrichum`
3. `leaf_healthy`
4. `leaf_phomopsis`
5. `leaf_rhizoctonia`

## Visibility policy

- `EXCLUDE_UNLEARNABLE`: visible projected short side below 2 pixels.
- `EXCLUDE_AMBIGUOUS`: different-class boxes overlap at IoU >= 0.95.
- `EXCLUDE_PROJECTION_ARTIFACT`: a projection covers over 90% of the frame while clipped on multiple sides.
- `REVIEWED_KEEP`: difficult AR>10, >50% area, or border case retained only when the rendered leaf remains visually identifiable in the all-frame contact-sheet review.
- `KEEP`: no policy flag.

Frames with a strong cross-split pHash distance <=4 are excluded from official val/test and preserved under `auxiliary/excluded_gazebo/`.

No coordinate is silently clipped or repaired. Every removed row is recorded in `manifests/excluded_samples.json`; all 5956 original rows have a decision in `gazebo_label_review.json`.

Decision counts: `{'EXCLUDE_AMBIGUOUS': 4, 'EXCLUDE_PROJECTION_ARTIFACT': 1, 'EXCLUDE_UNLEARNABLE': 10, 'KEEP': 5238, 'REVIEWED_KEEP': 703}`.

## Split policy

Versions are not source groups. v2/v3 samples with the same numbered tree share one canonical group. Tree 1, tree 2, and legacy g1 are train; tree 3 is validation; tree 4 is test. The shared static simulator world/camera background is recorded as a limitation and all strong perceptual candidates are independently audited.

Top-level and `domains/gazebo` official files are hard-linked to avoid duplicate storage. Deleting one directory entry does not alter the other inode entry.

## Real status

`REAL DETECTION GROUND TRUTH NOT AVAILABLE`. Do not create `real.yaml` until deployment-like Real images have independently reviewed individual-leaf boxes and group-disjoint splits.
