#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


CLASSES = [
    "leaf_algal",
    "leaf_blight",
    "leaf_colletotrichum",
    "leaf_healthy",
    "leaf_phomopsis",
    "leaf_rhizoctonia",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")
V3_SPLITS = {
    "durian_tree_1": "train",
    "durian_tree_2": "train",
    "durian_tree_3": "val",
    "durian_tree_4": "test",
}
V2_SPLITS = {
    "v2_g1": "train",
    "v2_tree_1": "train",
    "v2_tree_2": "val",
    "v2_tree_3": "test",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_link(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def image_for_manifest(root: Path, split: str, sample_id: str) -> Path:
    matches = [
        path
        for path in (root / "images" / split).glob(f"*{sample_id}.*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve image for {sample_id}: {matches}")
    return matches[0]


def label_for_manifest(root: Path, split: str, sample_id: str) -> Path:
    matches = list((root / "labels" / split).glob(f"*{sample_id}.txt"))
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve label for {sample_id}: {matches}")
    return matches[0]


def normalized_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    output = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) != 5:
            raise ValueError(f"Invalid label columns: {path}:{line_number}")
        raw_class, x, y, width, height = map(float, tokens)
        class_id = int(raw_class)
        if raw_class != class_id or class_id not in range(len(CLASSES)):
            raise ValueError(f"Invalid class: {path}:{line_number}")
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"Non-finite label: {path}:{line_number}")
        left = min(1.0, max(0.0, x - width / 2.0))
        right = min(1.0, max(0.0, x + width / 2.0))
        top = min(1.0, max(0.0, y - height / 2.0))
        bottom = min(1.0, max(0.0, y + height / 2.0))
        if right <= left or bottom <= top:
            raise ValueError(f"Empty label: {path}:{line_number}")
        record = (
            class_id,
            (left + right) / 2.0,
            (top + bottom) / 2.0,
            right - left,
            bottom - top,
        )
        rounded = tuple(round(value, 10) for value in record)
        if rounded not in seen:
            seen.add(rounded)
            output.append(record)
    if not output:
        raise ValueError(f"Empty label file: {path}")
    return output


def collect_v3(root: Path) -> list[dict]:
    rows = json.loads((root / "manifest.json").read_text())
    records = []
    for row in rows:
        group = f"v3_{row['source_group']}"
        split = V3_SPLITS.get(row["source_group"])
        if split is None or split != row["split"]:
            raise ValueError(f"Invalid v3 split: {row}")
        image = image_for_manifest(root, split, row["sample_id"])
        label = label_for_manifest(root, split, row["sample_id"])
        records.append(
            {
                "id": f"v3_{row['sample_id']}",
                "domain": "gazebo",
                "source": "gazebo_v3",
                "group": group,
                "split": split,
                "image": image,
                "source_label": label,
                "labels": normalized_labels(label),
                "sha256": sha256(image),
            }
        )
    return records


def v2_group(stem: str) -> str:
    if "gazebo_g1_" in stem:
        return "v2_g1"
    match = re.search(r"durian_tree_(\d+)", stem)
    if not match:
        raise ValueError(f"Cannot derive v2 group: {stem}")
    return f"v2_tree_{match.group(1)}"


def collect_v2(root: Path) -> list[dict]:
    records = []
    for old_split in ("train", "test", "val"):
        image_root = root / "images" / old_split
        if not image_root.exists():
            continue
        for image in sorted(image_root.iterdir()):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            source_label = root / "labels" / old_split / f"{image.stem}.txt"
            group = v2_group(image.stem)
            split = V2_SPLITS.get(group)
            if split is None:
                raise ValueError(f"No split for {group}")
            records.append(
                {
                    "id": f"v2_{image.stem}",
                    "domain": "gazebo",
                    "source": "gazebo_v2",
                    "group": group,
                    "split": split,
                    "image": image,
                    "source_label": source_label,
                    "labels": normalized_labels(source_label),
                    "sha256": sha256(image),
                }
            )
    return records


def deduplicate_gazebo(v3: list[dict], v2: list[dict]) -> tuple[list[dict], list[dict]]:
    seen = set()
    accepted = []
    rejected = []
    for record in v3 + v2:
        if record["sha256"] in seen:
            rejected.append(record)
            continue
        seen.add(record["sha256"])
        accepted.append(record)
    return accepted, rejected


def collect_real(root: Path) -> list[dict]:
    rows = json.loads((root / "manifest.json").read_text())
    output = []
    for row in rows:
        image = Path(row["image"])
        if not image.is_file():
            raise FileNotFoundError(image)
        class_id = int(row["class_id"])
        if row["class_name"] != CLASSES[class_id]:
            raise ValueError(f"Real class mismatch: {row}")
        output.append(
            {
                "id": f"real_{len(output):05d}",
                "domain": "real",
                "source": row["source"],
                "group": row["group"],
                "split": row["split"],
                "class_id": class_id,
                "image": image,
                "sha256": sha256(image),
            }
        )
    return output


def balanced_real_subset(records: list[dict], split: str, count: int, rng: random.Random) -> list[dict]:
    bins = defaultdict(list)
    for record in records:
        if record["split"] == split:
            bins[record["class_id"]].append(record)
    for class_id in range(len(CLASSES)):
        if not bins[class_id]:
            raise ValueError(f"No real records for {split} class {class_id}")
        rng.shuffle(bins[class_id])
    selected = []
    offsets = [0] * len(CLASSES)
    for index in range(count):
        class_id = index % len(CLASSES)
        values = bins[class_id]
        selected.append(values[offsets[class_id] % len(values)])
        offsets[class_id] += 1
    rng.shuffle(selected)
    return selected


def write_labels(path: Path, labels: list[tuple[int, float, float, float, float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{class_id} {x:.8f} {y:.8f} {width:.8f} {height:.8f}\n"
            for class_id, x, y, width, height in labels
        )
    )


def materialize_gazebo(root: Path, records: list[dict]) -> list[dict]:
    manifest = []
    for index, record in enumerate(records):
        name = f"gazebo_{index:04d}{record['image'].suffix.lower()}"
        image_out = root / "images" / record["split"] / name
        label_out = root / "labels" / record["split"] / f"{Path(name).stem}.txt"
        safe_link(record["image"], image_out)
        write_labels(label_out, record["labels"])
        manifest.append(
            {
                "id": record["id"],
                "domain": "gazebo",
                "source": record["source"],
                "group": record["group"],
                "split": record["split"],
                "image": str(image_out.resolve()),
                "label": str(label_out.resolve()),
                "sha256": record["sha256"],
                "boxes": len(record["labels"]),
            }
        )
    return manifest


def materialize_real_individual(root: Path, records: list[dict], split: str, prefix: str) -> list[dict]:
    manifest = []
    for index, record in enumerate(records):
        extension = record["image"].suffix.lower()
        name = f"{prefix}_{index:04d}{extension}"
        image_out = root / "images" / split / name
        label_out = root / "labels" / split / f"{Path(name).stem}.txt"
        safe_link(record["image"], image_out)
        write_labels(label_out, [(record["class_id"], 0.5, 0.5, 0.98, 0.98)])
        manifest.append(
            {
                "id": record["id"],
                "domain": "real",
                "source": record["source"],
                "group": record["group"],
                "split": split,
                "class_id": record["class_id"],
                "image": str(image_out.resolve()),
                "label": str(label_out.resolve()),
                "sha256": record["sha256"],
                "boxes": 1,
            }
        )
    return manifest


def class_schedule(total: int, rng: random.Random) -> list[int]:
    values = []
    for index in range(total):
        values.append(index % len(CLASSES))
    rng.shuffle(values)
    return values


def source_queues(records: list[dict], rng: random.Random) -> dict[int, deque]:
    bins = defaultdict(list)
    for record in records:
        if record["split"] == "train":
            bins[record["class_id"]].append(record)
    queues = {}
    for class_id in range(len(CLASSES)):
        rng.shuffle(bins[class_id])
        queues[class_id] = deque(bins[class_id])
    return queues


def next_source(queues: dict[int, deque], class_id: int, rng: random.Random) -> dict:
    queue = queues[class_id]
    if not queue:
        raise ValueError(f"No real source for class {class_id}")
    record = queue[0]
    queue.rotate(-1)
    return record


def build_real_collages(
    root: Path,
    real_records: list[dict],
    image_count: int,
    box_count: int,
    seed: int,
) -> list[dict]:
    if image_count <= 0 or box_count < image_count:
        raise ValueError("Invalid collage targets")
    rng = random.Random(seed)
    queues = source_queues(real_records, rng)
    schedule = class_schedule(box_count, rng)
    base, remainder = divmod(box_count, image_count)
    per_image = [base + (index < remainder) for index in range(image_count)]
    if max(per_image) > 25:
        raise ValueError(f"Too many objects per collage: {max(per_image)}")
    manifest = []
    cursor = 0
    for image_index, objects in enumerate(per_image):
        width, height = 1280, 960
        background = (
            rng.randint(42, 88),
            rng.randint(70, 118),
            rng.randint(35, 72),
        )
        canvas = Image.new("RGB", (width, height), background)
        columns = 5
        rows = max(1, math.ceil(objects / columns))
        cell_width = width // columns
        cell_height = height // rows
        labels = []
        source_groups = []
        for slot, class_id in enumerate(schedule[cursor:cursor + objects]):
            record = next_source(queues, class_id, rng)
            source_groups.append(record["group"])
            with Image.open(record["image"]) as source_image:
                tile = source_image.convert("RGB")
            if rng.random() < 0.5:
                tile = ImageOps.mirror(tile)
            tile = ImageEnhance.Brightness(tile).enhance(rng.uniform(0.82, 1.18))
            tile = ImageEnhance.Contrast(tile).enhance(rng.uniform(0.85, 1.15))
            max_width = max(8, cell_width - 12)
            max_height = max(8, cell_height - 12)
            tile.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            column = slot % columns
            row = slot // columns
            x0 = column * cell_width + rng.randint(4, max(4, cell_width - tile.width - 4))
            y0 = row * cell_height + rng.randint(4, max(4, cell_height - tile.height - 4))
            canvas.paste(tile, (x0, y0))
            labels.append(
                (
                    class_id,
                    (x0 + tile.width / 2.0) / width,
                    (y0 + tile.height / 2.0) / height,
                    tile.width / width,
                    tile.height / height,
                )
            )
        cursor += objects
        name = f"real_collage_{image_index:04d}.jpg"
        image_out = root / "images" / "train" / name
        label_out = root / "labels" / "train" / f"{Path(name).stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(image_out, quality=92, optimize=True)
        write_labels(label_out, labels)
        manifest.append(
            {
                "id": f"real_collage_{image_index:04d}",
                "domain": "real",
                "source": "clean_real_collage",
                "group": f"real_collage_{image_index:04d}",
                "source_groups": sorted(set(source_groups)),
                "split": "train",
                "image": str(image_out.resolve()),
                "label": str(label_out.resolve()),
                "sha256": sha256(image_out),
                "boxes": len(labels),
            }
        )
    if cursor != box_count:
        raise RuntimeError(f"Collage schedule mismatch: {cursor} != {box_count}")
    return manifest


def copy_gazebo_into_hybrid(original_manifest: list[dict], hybrid_root: Path) -> list[dict]:
    manifest = []
    for row in original_manifest:
        image = Path(row["image"])
        label = Path(row["label"])
        image_out = hybrid_root / "images" / row["split"] / image.name
        label_out = hybrid_root / "labels" / row["split"] / label.name
        safe_link(image, image_out)
        safe_link(label, label_out)
        manifest.append({**row, "image": str(image_out.resolve()), "label": str(label_out.resolve())})
    return manifest


def write_yaml(path: Path, dataset_root: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    workspace = Path(__file__).resolve().parents[2]
    try:
        portable_root = dataset_root.resolve().relative_to(workspace)
    except ValueError:
        portable_root = dataset_root.resolve()
    path.write_text(
        f"path: {portable_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names}\n"
    )


def audit_view(root: Path, manifest: list[dict], require_domain_balance: bool) -> dict:
    issues = []
    image_counts = Counter()
    box_counts = Counter()
    class_counts = Counter()
    size_counts = Counter()
    hashes = defaultdict(set)
    groups = defaultdict(set)
    for row in manifest:
        image_path = Path(row["image"])
        label_path = Path(row["label"])
        if not image_path.is_file() or not label_path.is_file():
            issues.append(f"missing_pair:{image_path}:{label_path}")
            continue
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
        digest = sha256(image_path)
        hashes[digest].add(row["split"])
        groups[(row["domain"], row["group"])].add(row["split"])
        labels = normalized_labels(label_path)
        if len(labels) != row["boxes"]:
            issues.append(f"box_count_mismatch:{label_path}")
        image_counts[(row["split"], row["domain"])] += 1
        box_counts[(row["split"], row["domain"])] += len(labels)
        for class_id, _, _, box_width, box_height in labels:
            class_counts[(row["split"], row["domain"], class_id)] += 1
            short = min(box_width * width, box_height * height)
            size = "small_under_32" if short < 32 else "medium_32_95" if short < 96 else "large_96_plus"
            size_counts[(row["split"], row["domain"], size)] += 1
    issues.extend(
        f"hash_leak:{digest}:{sorted(splits)}"
        for digest, splits in hashes.items()
        if len(splits) > 1
    )
    issues.extend(
        f"group_leak:{key}:{sorted(splits)}"
        for key, splits in groups.items()
        if len(splits) > 1
    )
    if require_domain_balance:
        for split in SPLITS:
            real_images = image_counts[(split, "real")]
            gazebo_images = image_counts[(split, "gazebo")]
            if real_images != gazebo_images:
                issues.append(f"domain_image_imbalance:{split}:{real_images}:{gazebo_images}")
        real_boxes = box_counts[("train", "real")]
        gazebo_boxes = box_counts[("train", "gazebo")]
        if real_boxes != gazebo_boxes:
            issues.append(f"domain_box_imbalance:train:{real_boxes}:{gazebo_boxes}")
    encode = lambda values: {
        "|".join(map(str, key)): value for key, value in sorted(values.items())
    }
    return {
        "pass": not issues,
        "issues": issues,
        "images": len(manifest),
        "boxes": sum(box_counts.values()),
        "image_counts": encode(image_counts),
        "box_counts": encode(box_counts),
        "class_counts": encode(class_counts),
        "size_counts": encode(size_counts),
    }


def write_readme(root: Path, gazebo_records: list[dict], rejected: list[dict], audit: dict):
    root.joinpath("README.md").write_text(
        "# Hybrid Durian Leaf Dataset\n\n"
        "This is the only training dataset used by the three six-class YOLO experiments.\n\n"
        "- `original/`: Gazebo-only, group-disjoint train/validation/test splits.\n"
        "- `hybrid/`: the same Gazebo samples plus clean real-leaf data. Training is balanced "
        "by both domain image count and domain bounding-box count.\n"
        "- `evaluation/`: separate Gazebo and real-domain evaluation YAML files.\n"
        "- `audit.json`: integrity, leakage, balance, class, and object-size checks.\n\n"
        "Real training collages are deterministic small-object augmentations generated only from "
        "clean real training groups. Real validation and test images remain unmodified and group-disjoint.\n\n"
        f"Gazebo source images accepted: {len(gazebo_records)}\n\n"
        f"Exact duplicate Gazebo images removed: {len(rejected)}\n\n"
        f"Audit pass: {audit['pass']}\n"
    )


def arguments():
    workspace = Path("/home/johny/durian_ws")
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", type=Path, default=workspace / "hybrid_datasets/hybrid_v3_clean/gazebo")
    parser.add_argument("--v2", type=Path, default=workspace / "hybrid_datasets/hybrid_v2/gazebo_dataset")
    parser.add_argument(
        "--real",
        type=Path,
        default=workspace / "hybrid_datasets/hybrid_v4_two_stage/classifier",
    )
    parser.add_argument("--output", type=Path, default=workspace / "experiment/hybrid_dataset")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    gazebo, rejected = deduplicate_gazebo(collect_v3(args.v3), collect_v2(args.v2))
    real = collect_real(args.real)
    rng = random.Random(args.seed)

    original_root = args.output / "original"
    original_manifest = materialize_gazebo(original_root, gazebo)
    write_yaml(original_root / "data.yaml", original_root)
    original_root.joinpath("manifest.json").write_text(json.dumps(original_manifest, indent=2))

    hybrid_root = args.output / "hybrid"
    hybrid_manifest = copy_gazebo_into_hybrid(original_manifest, hybrid_root)
    gazebo_train_images = sum(row["split"] == "train" for row in original_manifest)
    gazebo_train_boxes = sum(
        row["boxes"] for row in original_manifest if row["split"] == "train"
    )
    hybrid_manifest.extend(
        build_real_collages(
            hybrid_root,
            real,
            gazebo_train_images,
            gazebo_train_boxes,
            args.seed,
        )
    )
    for split in ("val", "test"):
        gazebo_count = sum(row["split"] == split for row in original_manifest)
        selected = balanced_real_subset(real, split, gazebo_count, rng)
        hybrid_manifest.extend(
            materialize_real_individual(hybrid_root, selected, split, f"real_{split}")
        )
    write_yaml(hybrid_root / "data.yaml", hybrid_root)
    hybrid_root.joinpath("manifest.json").write_text(json.dumps(hybrid_manifest, indent=2))

    evaluation_root = args.output / "evaluation"
    write_yaml(evaluation_root / "gazebo.yaml", original_root)
    real_eval_root = evaluation_root / "real"
    real_eval_manifest = []
    for split in ("val", "test"):
        selected_rows = [
            row for row in hybrid_manifest
            if row["domain"] == "real" and row["split"] == split
        ]
        for index, row in enumerate(selected_rows):
            image = Path(row["image"])
            label = Path(row["label"])
            image_out = real_eval_root / "images" / split / image.name
            label_out = real_eval_root / "labels" / split / label.name
            safe_link(image, image_out)
            safe_link(label, label_out)
            real_eval_manifest.append(
                {**row, "image": str(image_out.resolve()), "label": str(label_out.resolve())}
            )
    train_seed = balanced_real_subset(real, "train", len(CLASSES), rng)
    real_eval_manifest.extend(
        materialize_real_individual(real_eval_root, train_seed, "train", "real_train_seed")
    )
    write_yaml(evaluation_root / "real.yaml", real_eval_root)
    real_eval_root.joinpath("manifest.json").write_text(json.dumps(real_eval_manifest, indent=2))

    audit = {
        "schema_version": 5,
        "seed": args.seed,
        "classes": CLASSES,
        "sources": {
            "gazebo_v3": str(args.v3.resolve()),
            "gazebo_v2": str(args.v2.resolve()),
            "clean_real": str(args.real.resolve()),
            "gazebo_exact_duplicates_removed": [record["id"] for record in rejected],
        },
        "original": audit_view(original_root, original_manifest, False),
        "hybrid": audit_view(hybrid_root, hybrid_manifest, True),
        "real_evaluation": audit_view(real_eval_root, real_eval_manifest, False),
    }
    audit["pass"] = all(
        audit[key]["pass"] for key in ("original", "hybrid", "real_evaluation")
    )
    args.output.joinpath("audit.json").write_text(json.dumps(audit, indent=2))
    write_readme(args.output, gazebo, rejected, audit)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "pass": audit["pass"],
                "original": {
                    "images": audit["original"]["images"],
                    "boxes": audit["original"]["boxes"],
                },
                "hybrid": {
                    "images": audit["hybrid"]["images"],
                    "boxes": audit["hybrid"]["boxes"],
                },
            },
            indent=2,
        )
    )
    if not audit["pass"]:
        raise SystemExit("Dataset audit failed; inspect audit.json")


if __name__ == "__main__":
    main()
