# Dataset Diagnosis

## Dataset validity

The official dataset is technically valid for its stated **Gazebo-only** scope: 252 images and 5886 reviewed boxes. Real data is deliberately unlabelled.

## Annotation validity

All original 5,956 Gazebo annotations received an explicit decision. 15 were excluded under the visibility policy; 5886 remain. The retained labels denote visible rendered leaf planes. They do not prove physical-Real annotation validity.

## Split independence

Exact, decoded-pixel, canonical source-group, and same-tree/camera-family leakage are zero. Strong perceptual candidates were investigated and are caused by the shared static synthetic background across different target trees.

## Class balance and object distribution

| Split | Images | Boxes | Boxes/image mean/median/min/max | Class counts 0..5 | Symmetric AR>10 |
|---|---:|---:|---|---|---:|
| train | 144 | 3412 | 23.69/25/10/38 | [292, 839, 658, 575, 654, 394] | 135 |
| val | 62 | 1516 | 24.45/26/9/33 | [267, 299, 159, 162, 377, 252] | 51 |
| test | 46 | 958 | 20.83/21/9/31 | [133, 138, 47, 266, 274, 100] | 38 |

Full area, short-side, aspect, and class distributions are recorded in `integrity_report.md` and `manifests/audit.json`.

## Domain balance and Real availability

Official detector GT is 100% Gazebo. 141 whole Real photographs are preserved as unlabelled sources; 120 legacy collages are preserved as auxiliary images without labels. Valid Real training, validation, and final-test GT are unavailable.

## Deployment similarity

The data matches the repository's Gazebo runtime appearance but does not establish physical-orchard performance. The shared world/background and flat rendered leaf planes remain limitations.

## Architecture suitability

- YOLO26n: suitable for a Gazebo-only controlled benchmark.
- YOLO26n-P2: suitable and potentially useful for the retained small objects.
- RF-DETR-N: suitable for the same Gazebo-only benchmark.
- D-FINE-N: suitable for the same Gazebo-only benchmark.
- None is validated for the full deployed Real+Gazebo task until Real individual-leaf GT exists.

## Explicit answers

1. Is the new dataset technically valid? **Yes, for the documented Gazebo-only scope.**
2. Are annotations semantically valid? **Yes for reviewed rendered Gazebo leaf planes; Real is unavailable.**
3. Is Gazebo clean enough for training? **Yes, with documented synthetic limitations.**
4. Is Gazebo val independent? **Yes by tree/source group and same-tree camera family.**
5. Is Gazebo test independent? **Yes by tree/source group and same-tree camera family.**
6. Is valid Real detection training data available? **No.**
7. Is valid Real validation available? **No.**
8. Is valid Real final testing available? **No.**
9. Is cross-split leakage zero? **Known exact, decoded, source-group, and annotated-tree leakage is zero; shared-world perceptual similarity is documented.**
10. Can it fairly compare architectures? **Yes for a controlled Gazebo-only comparison; no for full deployment claims.**
11. What limitations remain? **No Real GT, one shared synthetic world, limited independent trees, flat leaf-plane rendering, and residual difficult projection geometry.**
12. Is retraining now justified? **Only for Gazebo-only development/benchmarking, not final Real deployment validation.**
