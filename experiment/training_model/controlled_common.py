#!/usr/bin/env python3
"""Shared, dependency-free controls for the frozen detector study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets"
CONTROL = ROOT / "controlled_experiment"
SPLITS = ("train", "val", "test")
NAMES = [
    "leaf_algal", "leaf_blight", "leaf_colletotrichum",
    "leaf_healthy", "leaf_phomopsis", "leaf_rhizoctonia",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def dataset_records() -> list[dict]:
    records = []
    image_id = 1
    annotation_id = 1
    for split in SPLITS:
        images = sorted((DATASET / "images" / split).glob("*"))
        labels = {p.stem: p for p in (DATASET / "labels" / split).glob("*.txt")}
        for image in images:
            label = labels.get(image.stem)
            if label is None:
                raise ValueError(f"missing label for {image}")
            # PNG/JPEG dimensions without requiring OpenCV/Pillow.
            raw = image.read_bytes()
            if raw[:8] == b"\x89PNG\r\n\x1a\n":
                width = int.from_bytes(raw[16:20], "big")
                height = int.from_bytes(raw[20:24], "big")
            else:
                from PIL import Image
                with Image.open(image) as decoded:
                    width, height = decoded.size
            boxes = []
            for line_no, line in enumerate(label.read_text().splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"invalid YOLO row {label}:{line_no}")
                class_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                if class_id not in range(len(NAMES)) or not all(0 <= v <= 1 for v in (xc, yc, bw, bh)):
                    raise ValueError(f"invalid class/coordinate {label}:{line_no}")
                x1, y1 = (xc - bw / 2) * width, (yc - bh / 2) * height
                x2, y2 = (xc + bw / 2) * width, (yc + bh / 2) * height
                if not (x2 > x1 and y2 > y1 and x1 >= -1e-6 and y1 >= -1e-6 and x2 <= width + 1e-6 and y2 <= height + 1e-6):
                    raise ValueError(f"out-of-bounds box {label}:{line_no}")
                boxes.append({
                    "annotation_id": annotation_id,
                    "line": line_no,
                    "class_id": class_id,
                    "category_id": class_id + 1,
                    "yolo": [xc, yc, bw, bh],
                    "xyxy": [x1, y1, x2, y2],
                })
                annotation_id += 1
            records.append({
                "image_id": image_id, "split": split, "file_name": image.name,
                "image_path": image.relative_to(DATASET).as_posix(),
                "label_path": label.relative_to(DATASET).as_posix(),
                "width": width, "height": height,
                "image_sha256": sha256(image), "label_sha256": sha256(label),
                "boxes": boxes,
            })
            image_id += 1
    return records
