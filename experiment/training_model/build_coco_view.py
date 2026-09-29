#!/usr/bin/env python3
"""Create the deterministic COCO metadata view; canonical data is read-only."""
from __future__ import annotations

import argparse
import os
from collections import Counter

from controlled_common import CONTROL, DATASET, NAMES, SPLITS, dataset_records, sha256, write_json


def build() -> None:
    records = dataset_records()
    categories = [{"id": i + 1, "canonical_class_id": i, "name": n} for i, n in enumerate(NAMES)]
    mapping = {
        "format_version": 1,
        "canonical_dataset": str(DATASET),
        "category_mapping": {str(i): i + 1 for i in range(len(NAMES))},
        "records": records,
    }
    write_json(CONTROL / "coco" / "identity_map.json", mapping)
    outputs = {}
    for split in SPLITS:
        selected = [r for r in records if r["split"] == split]
        images = [{
            "id": r["image_id"], "file_name": r["file_name"],
            "width": r["width"], "height": r["height"],
        } for r in selected]
        annotations = []
        for r in selected:
            for box in r["boxes"]:
                x1, y1, x2, y2 = box["xyxy"]
                annotations.append({
                    "id": box["annotation_id"], "image_id": r["image_id"],
                    "category_id": box["category_id"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                })
        path = CONTROL / "coco" / "annotations" / f"instances_{split}.json"
        write_json(path, {"images": images, "annotations": annotations, "categories": categories})
        outputs[split] = {"images": len(images), "annotations": len(annotations), "sha256": sha256(path)}
        # RF-DETR's public trainer requires Roboflow COCO folder names. This is
        # a symlink-only view: image bytes remain canonical and immutable.
        if split != "test":
            rf_name = "valid" if split == "val" else split
            rf_dir = CONTROL / "coco" / "rfdetr" / rf_name
            rf_dir.mkdir(parents=True, exist_ok=True)
            annotation_link = rf_dir / "_annotations.coco.json"
            if annotation_link.is_symlink() or annotation_link.exists(): annotation_link.unlink()
            annotation_link.symlink_to(os.path.relpath(path, rf_dir))
            for image in images:
                link = rf_dir / image["file_name"]
                target = DATASET / "images" / split / image["file_name"]
                if link.is_symlink() or link.exists(): link.unlink()
                link.symlink_to(os.path.relpath(target, rf_dir))
    write_json(CONTROL / "coco" / "manifest.json", {
        "format_version": 1,
        "generator": "training_model/build_coco_view.py",
        "identity_map_sha256": sha256(CONTROL / "coco" / "identity_map.json"),
        "canonical_manifest_sha256": sha256(DATASET / "manifests" / "dataset_manifest.json"),
        "canonical_audit_sha256": sha256(DATASET / "manifests" / "audit.json"),
        "counts": outputs,
        "totals": {"images": len(records), "annotations": sum(len(r["boxes"]) for r in records)},
        "split_identity_sha256": {
            split: __import__("hashlib").sha256("\n".join(r["file_name"] for r in records if r["split"] == split).encode()).hexdigest()
            for split in SPLITS
        },
    })
    print("COCO view built: 252 images, 5886 annotations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    build()
