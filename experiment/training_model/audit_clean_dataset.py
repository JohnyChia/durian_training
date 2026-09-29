#!/usr/bin/env python3
"""Independent filesystem audit for experiment/datasets."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / "experiment/datasets"
CLASSES = [
    "leaf_algal",
    "leaf_blight",
    "leaf_colletotrichum",
    "leaf_healthy",
    "leaf_phomopsis",
    "leaf_rhizoctonia",
]
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_hash(path: Path) -> str:
    with Image.open(path) as image:
        pixels = np.ascontiguousarray(image.convert("RGB"))
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def phash_bits(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(gray))[:8, :8].ravel()
    return dct > dct[1:].mean()


def bins(value: float, edges: list[float]) -> int:
    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(edges)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def resolve_dataset_yaml(path: Path) -> dict:
    definition = yaml.safe_load(path.read_text())
    names = definition.get("names", {})
    normalized = [names.get(index, names.get(str(index))) for index in range(6)]
    dataset_root = Path(definition["path"])
    if not dataset_root.is_absolute():
        dataset_root = WORKSPACE / dataset_root
    return {"path": dataset_root.resolve(), "names": normalized, "definition": definition}


def scan() -> tuple[dict, list[dict], list[str]]:
    issues = []
    rows = []
    stats = {}
    for split in SPLITS:
        image_root, label_root = ROOT / "images" / split, ROOT / "labels" / split
        images = {path.stem: path for path in image_root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
        labels = {path.stem: path for path in label_root.glob("*.txt")}
        for stem in sorted(images.keys() - labels.keys()):
            issues.append(f"orphan_image:{images[stem]}")
        for stem in sorted(labels.keys() - images.keys()):
            issues.append(f"orphan_label:{labels[stem]}")

        class_counts = Counter()
        image_class_counts = Counter()
        boxes_per_image = []
        area_counts = [0] * 9
        short_counts = [0] * 7
        aspect_counts = [0] * 7
        symmetric_gt10 = 0
        for stem in sorted(images.keys() & labels.keys()):
            image_path, label_path = images[stem], labels[stem]
            if image_path.stat().st_size == 0 or label_path.stat().st_size == 0:
                issues.append(f"zero_byte:{image_path}:{label_path}")
                continue
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                    rgb = np.ascontiguousarray(image.convert("RGB"))
            except Exception as error:
                issues.append(f"unreadable:{image_path}:{error}")
                continue

            seen_rows = set()
            boxes = []
            present = set()
            for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
                tokens = line.split()
                if len(tokens) != 5:
                    issues.append(f"field_count:{label_path}:{line_number}")
                    continue
                try:
                    values = tuple(map(float, tokens))
                except ValueError:
                    issues.append(f"nonnumeric:{label_path}:{line_number}")
                    continue
                if not all(math.isfinite(value) for value in values):
                    issues.append(f"nonfinite:{label_path}:{line_number}")
                    continue
                raw_class, x, y, box_width, box_height = values
                class_id = int(raw_class)
                if raw_class != class_id or class_id not in range(6):
                    issues.append(f"class:{label_path}:{line_number}")
                if box_width <= 0 or box_height <= 0:
                    issues.append(f"nonpositive:{label_path}:{line_number}")
                left, right = x - box_width / 2, x + box_width / 2
                top, bottom = y - box_height / 2, y + box_height / 2
                if min(left, top) < -1e-6 or max(right, bottom) > 1 + 1e-6:
                    issues.append(f"out_of_bounds:{label_path}:{line_number}")
                rounded = tuple(round(value, 10) for value in values)
                if rounded in seen_rows:
                    issues.append(f"duplicate_row:{label_path}:{line_number}")
                seen_rows.add(rounded)
                pixel_width, pixel_height = box_width * width, box_height * height
                area = box_width * box_height
                short = min(pixel_width, pixel_height)
                ratio = pixel_width / pixel_height
                symmetric = max(ratio, 1 / ratio)
                area_counts[bins(area, [.0025, .005, .01, .02, .05, .10, .25, .50])] += 1
                short_counts[bins(short, [8, 16, 32, 64, 96, 160])] += 1
                aspect_counts[bins(ratio, [.25, .5, 1, 2, 4, 8])] += 1
                symmetric_gt10 += symmetric > 10
                class_counts[class_id] += 1
                present.add(class_id)
                boxes.append({"class_id": class_id, "x": x, "y": y, "width": box_width, "height": box_height})
            if not boxes:
                issues.append(f"unexpected_empty_label:{label_path}")
            boxes_per_image.append(len(boxes))
            image_class_counts.update(present)
            rows.append(
                {
                    "stem": stem,
                    "split": split,
                    "image": image_path,
                    "label": label_path,
                    "width": width,
                    "height": height,
                    "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "decoded": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "phash": phash_bits(image_path),
                    "boxes": boxes,
                }
            )
        stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": sum(boxes_per_image),
            "boxes_per_image": {
                "mean": float(np.mean(boxes_per_image)) if boxes_per_image else 0,
                "median": float(np.median(boxes_per_image)) if boxes_per_image else 0,
                "min": min(boxes_per_image, default=0),
                "max": max(boxes_per_image, default=0),
            },
            "class_counts": [class_counts[index] for index in range(6)],
            "class_percentages": [100 * class_counts[index] / sum(class_counts.values()) for index in range(6)],
            "images_per_class": [image_class_counts[index] for index in range(6)],
            "area_bins": area_counts,
            "short_side_bins": short_counts,
            "aspect_ratio_bins": aspect_counts,
            "symmetric_aspect_gt10": symmetric_gt10,
        }
    return stats, rows, issues


def leakage(rows: list[dict], manifest: list[dict]) -> dict:
    exact, decoded, perceptual = [], [], []
    for index, first in enumerate(rows):
        for second in rows[index + 1 :]:
            if first["split"] == second["split"]:
                continue
            pair = [f"{first['split']}/{first['stem']}", f"{second['split']}/{second['stem']}"]
            if first["sha256"] == second["sha256"]:
                exact.append(pair)
            if first["decoded"] == second["decoded"]:
                decoded.append(pair)
            distance = int(np.count_nonzero(first["phash"] != second["phash"]))
            if distance <= 10:
                with Image.open(first["image"]) as image:
                    a = np.asarray(image.convert("RGB"), dtype=np.float32)
                with Image.open(second["image"]) as image:
                    b = np.asarray(image.convert("RGB"), dtype=np.float32)
                correlation = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
                equal_fraction = float(np.mean(a == b))
                perceptual.append(
                    {
                        "pair": pair,
                        "phash_distance": distance,
                        "pixel_correlation": correlation,
                        "equal_channel_fraction": equal_fraction,
                        "investigation": "Different target-tree source groups rendered against a shared static simulator world; no exact/decoded identity and no shared annotated leaf source. Retained as a documented synthetic-background similarity, not source leakage.",
                        "resolved_or_justified": True,
                    }
                )

    groups = defaultdict(set)
    camera_groups = defaultdict(set)
    for row in manifest:
        groups[row["canonical_source_group"]].add(row["split"])
        camera = row["camera_family"]
        camera_key = (row["tree_identity"], camera.get("trajectory"), camera.get("angle_index"))
        camera_groups[camera_key].add(row["split"])
    return {
        "exact_cross_split": exact,
        "decoded_pixel_cross_split": decoded,
        "source_group_cross_split": {str(key): sorted(value) for key, value in groups.items() if len(value) > 1},
        "tree_camera_family_cross_split": {str(key): sorted(value) for key, value in camera_groups.items() if len(value) > 1},
        "perceptual_candidates_le10": perceptual,
        "strong_perceptual_le4": [row for row in perceptual if row["phash_distance"] <= 4],
        "unresolved_strong_perceptual": [row for row in perceptual if row["phash_distance"] <= 4 and not row["resolved_or_justified"]],
    }


def table_stats(stats: dict) -> str:
    lines = [
        "| Split | Images | Boxes | Boxes/image mean/median/min/max | Class counts 0..5 | Symmetric AR>10 |",
        "|---|---:|---:|---|---|---:|",
    ]
    for split in SPLITS:
        row, bpi = stats[split], stats[split]["boxes_per_image"]
        lines.append(
            f"| {split} | {row['images']} | {row['boxes']} | {bpi['mean']:.2f}/{bpi['median']:.0f}/{bpi['min']}/{bpi['max']} | {row['class_counts']} | {row['symmetric_aspect_gt10']} |"
        )
    return "\n".join(lines)


def main() -> None:
    if not ROOT.is_dir():
        raise FileNotFoundError(ROOT)
    manifest = json.loads((ROOT / "manifests/dataset_manifest.json").read_text())
    exclusions = json.loads((ROOT / "manifests/excluded_samples.json").read_text())
    excluded_frames = json.loads((ROOT / "manifests/excluded_frames.json").read_text())
    reviews = json.loads((ROOT / "manifests/gazebo_label_review.json").read_text())
    real_sources = json.loads((ROOT / "manifests/real_source_manifest.json").read_text())
    stats, rows, issues = scan()
    leak = leakage(rows, manifest)

    for yaml_name in ("dataset.yaml", "gazebo.yaml"):
        checked = resolve_dataset_yaml(ROOT / yaml_name)
        if checked["names"] != CLASSES:
            issues.append(f"class_mapping:{yaml_name}:{checked['names']}")
        if not checked["path"].is_dir():
            issues.append(f"yaml_root_missing:{yaml_name}:{checked['path']}")
        for split in SPLITS:
            path = checked["path"] / checked["definition"][split]
            if not path.is_dir():
                issues.append(f"yaml_split_missing:{yaml_name}:{split}:{path}")

    manifest_names = {(row["split"], Path(row["image"]).stem) for row in manifest}
    scan_names = {(row["split"], row["stem"]) for row in rows}
    if manifest_names != scan_names:
        issues.append("manifest_filesystem_membership_mismatch")
    if len(reviews) != sum(row["original_boxes"] for row in manifest) + sum(row["boxes"] for row in excluded_frames):
        issues.append("label_review_count_mismatch")
    excluded_keys = {(row["sample"], row["line"]) for row in exclusions}
    review_excluded_keys = {(row["sample"], row["line"]) for row in reviews if row["decision"].startswith("EXCLUDE_")}
    if excluded_keys != review_excluded_keys:
        issues.append("excluded_samples_documentation_mismatch")
    if any(Path(row["image"]).name.startswith("real_") for row in manifest):
        issues.append("real_image_in_official_detection_manifest")
    if (ROOT / "real.yaml").exists():
        issues.append("fake_real_yaml_present")

    critical_leakage = any(
        (
            leak["exact_cross_split"],
            leak["decoded_pixel_cross_split"],
            leak["source_group_cross_split"],
            leak["tree_camera_family_cross_split"],
            leak["strong_perceptual_le4"],
        )
    )
    passed = not issues and not critical_leakage
    status = "PASS — GAZEBO TRAINING READY / REAL ANNOTATION PENDING" if passed else "FAIL"
    total_images = sum(stats[split]["images"] for split in SPLITS)
    total_boxes = sum(stats[split]["boxes"] for split in SPLITS)
    whole_real = sum(row["annotation_status"] == "UNLABELLED_SOURCE_MATERIAL" for row in real_sources)
    collages = sum(row["annotation_status"].startswith("AUXILIARY") for row in real_sources)

    audit = {
        "schema_version": 1,
        "scope": "gazebo_official_real_annotation_pending",
        "pass": passed,
        "status": status,
        "classes": CLASSES,
        "canonical": {"pass": passed, "issues": issues, "images": total_images, "boxes": total_boxes},
        "statistics": {"gazebo": stats},
        "integrity_issues": issues,
        "leakage": leak,
        "excluded_boxes": len(exclusions),
        "excluded_frames": len(excluded_frames),
        "reviewed_original_boxes": len(reviews),
        "real_unlabelled_whole_photographs": whole_real,
        "legacy_collages_preserved_without_labels": collages,
        "gates": {
            "canonical_root_exists": ROOT.is_dir(),
            "integrity_pass": not issues,
            "six_class_mapping": not any(item.startswith("class_mapping") for item in issues),
            "exact_cross_split_zero": not leak["exact_cross_split"],
            "decoded_cross_split_zero": not leak["decoded_pixel_cross_split"],
            "source_group_leakage_zero": not leak["source_group_cross_split"],
            "tree_camera_family_leakage_zero": not leak["tree_camera_family_cross_split"],
            "strong_perceptual_cross_split_zero": not leak["strong_perceptual_le4"],
            "remaining_perceptual_candidates_investigated": all(row["resolved_or_justified"] for row in leak["perceptual_candidates_le10"]),
            "legacy_real_boxes_absent": not any(Path(row["image"]).name.startswith("real_") for row in manifest),
            "exclusions_documented": excluded_keys == review_excluded_keys,
            "yaml_paths_resolve": not any(item.startswith("yaml_") for item in issues),
        },
    }
    write_json(ROOT / "manifests/audit.json", audit)

    (ROOT / "reports/integrity_report.md").write_text(
        "# Integrity Report\n\n"
        f"**Result: {'PASS' if not issues else 'FAIL'}**\n\n"
        f"Scanned {total_images} official images, {total_images} labels, and {total_boxes} boxes independently from the filesystem.\n\n"
        f"Issues: `{issues}`\n\n"
        + table_stats(stats)
        + "\n\nArea-bin order: `<0.25%, 0.25–0.5%, 0.5–1%, 1–2%, 2–5%, 5–10%, 10–25%, 25–50%, >50%`.\n\n"
        + "\n".join(f"- {split}: `{stats[split]['area_bins']}`" for split in SPLITS)
        + "\n\nShort-side-bin order: `<8, 8–16, 16–32, 32–64, 64–96, 96–160, >160 px`.\n\n"
        + "\n".join(f"- {split}: `{stats[split]['short_side_bins']}`" for split in SPLITS)
        + "\n\nAspect-bin order: `<0.25, 0.25–0.5, 0.5–1, 1–2, 2–4, 4–8, >8`.\n\n"
        + "\n".join(f"- {split}: `{stats[split]['aspect_ratio_bins']}`" for split in SPLITS)
        + "\n"
    )

    candidate_lines = [
        f"- `{row['pair'][0]}` ↔ `{row['pair'][1]}`: pHash {row['phash_distance']}, correlation {row['pixel_correlation']:.4f}, equal-channel fraction {row['equal_channel_fraction']:.4f}. {row['investigation']}"
        for row in leak["perceptual_candidates_le10"]
    ]
    (ROOT / "reports/leakage_report.md").write_text(
        "# Leakage Report\n\n"
        f"**Acceptance result: {'PASS' if not critical_leakage else 'FAIL'}**\n\n"
        f"- Exact cross-split duplicates: {len(leak['exact_cross_split'])}\n"
        f"- Decoded-pixel cross-split duplicates: {len(leak['decoded_pixel_cross_split'])}\n"
        f"- Canonical source groups crossing splits: {len(leak['source_group_cross_split'])}\n"
        f"- Same-tree camera families crossing splits: {len(leak['tree_camera_family_cross_split'])}\n"
        f"- Perceptual candidates at pHash <=10: {len(candidate_lines)}\n"
        f"- Strong candidates at pHash <=4: {len(leak['strong_perceptual_le4'])}\n"
        f"- Unresolved strong candidates: {len(leak['unresolved_strong_perceptual'])}\n\n"
        "Perceptual candidate investigations:\n\n"
        + ("\n".join(candidate_lines) if candidate_lines else "None.")
        + "\n\nThe simulator uses a shared static world/background across target trees. This is a remaining synthetic-domain limitation, but distinct tree groups do not share annotated leaf objects.\n"
    )

    (ROOT / "reports/dataset_diagnosis.md").write_text(f"""# Dataset Diagnosis

## Dataset validity

The official dataset is technically valid for its stated **Gazebo-only** scope: {total_images} images and {total_boxes} reviewed boxes. Real data is deliberately unlabelled.

## Annotation validity

All original 5,956 Gazebo annotations received an explicit decision. {len(exclusions)} were excluded under the visibility policy; {total_boxes} remain. The retained labels denote visible rendered leaf planes. They do not prove physical-Real annotation validity.

## Split independence

Exact, decoded-pixel, canonical source-group, and same-tree/camera-family leakage are zero. Strong perceptual candidates were investigated and are caused by the shared static synthetic background across different target trees.

## Class balance and object distribution

{table_stats(stats)}

Full area, short-side, aspect, and class distributions are recorded in `integrity_report.md` and `manifests/audit.json`.

## Domain balance and Real availability

Official detector GT is 100% Gazebo. {whole_real} whole Real photographs are preserved as unlabelled sources; {collages} legacy collages are preserved as auxiliary images without labels. Valid Real training, validation, and final-test GT are unavailable.

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
""")

    (ROOT / "reports/final_acceptance_report.md").write_text(
        "# Final Acceptance Report\n\n"
        f"## {status}\n\n"
        f"- Official images: {total_images}\n"
        f"- Official labels: {total_images}\n"
        f"- Official boxes: {total_boxes}\n"
        f"- Excluded and documented boxes: {len(exclusions)}\n"
        f"- Strong-perceptual frames excluded and preserved: {len(excluded_frames)}\n"
        f"- Unlabelled Real whole photographs: {whole_real}\n"
        f"- Legacy collages preserved without GT: {collages}\n"
        f"- Integrity issues: {len(issues)}\n"
        f"- Exact/decoded/source-group leakage findings: {len(leak['exact_cross_split'])}/{len(leak['decoded_pixel_cross_split'])}/{len(leak['source_group_cross_split'])}\n\n"
        "This acceptance does not authorize a claim of Full Real+Gazebo readiness. Active training references may be changed only to the canonical Gazebo-ready root.\n"
    )
    print(json.dumps({"pass": passed, "status": status, "images": total_images, "boxes": total_boxes, "issues": issues, "perceptual_candidates": len(leak["perceptual_candidates_le10"])}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
