#!/usr/bin/env python3
"""Prove YOLO/COCO identity, geometry, split isolation, and hashes."""
from __future__ import annotations

from controlled_common import CONTROL, NAMES, SPLITS, dataset_records, read_json, sha256


def verify() -> dict:
    expected = dataset_records()
    identity = read_json(CONTROL / "coco" / "identity_map.json")
    assert identity["records"] == expected, "identity map differs from canonical YOLO files"
    manifest = read_json(CONTROL / "coco" / "manifest.json")
    assert manifest["identity_map_sha256"] == sha256(CONTROL / "coco" / "identity_map.json")
    seen = set()
    totals = {"images": 0, "annotations": 0}
    by_id = {r["image_id"]: r for r in expected}
    for split in SPLITS:
        coco = read_json(CONTROL / "coco" / "annotations" / f"instances_{split}.json")
        assert coco["categories"] == [{"canonical_class_id": i, "id": i + 1, "name": n} for i, n in enumerate(NAMES)]
        assert manifest["counts"][split]["sha256"] == sha256(CONTROL / "coco" / "annotations" / f"instances_{split}.json")
        ids = {i["id"] for i in coco["images"]}
        assert not ids & seen, "split leakage"
        seen |= ids
        assert all(by_id[i]["split"] == split for i in ids)
        actual = {a["id"]: a for a in coco["annotations"]}
        expected_boxes = [b | {"image_id": r["image_id"]} for r in expected if r["split"] == split for b in r["boxes"]]
        assert len(actual) == len(expected_boxes)
        for box in expected_boxes:
            a = actual[box["annotation_id"]]
            x1, y1, x2, y2 = box["xyxy"]
            assert a["image_id"] == box["image_id"] and a["category_id"] == box["class_id"] + 1
            assert max(abs(x - y) for x, y in zip(a["bbox"], [x1, y1, x2 - x1, y2 - y1])) < 1e-9
        totals["images"] += len(coco["images"])
        totals["annotations"] += len(coco["annotations"])
    assert totals == {"images": 252, "annotations": 5886} and len(seen) == 252
    return totals


if __name__ == "__main__":
    result = verify()
    print(f"PASS: YOLO <-> COCO identity verified ({result['images']} images, {result['annotations']} boxes, no leakage)")
