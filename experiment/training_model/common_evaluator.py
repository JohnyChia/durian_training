#!/usr/bin/env python3
"""Framework-independent, COCO-style 101-point detection evaluator.

Predictions are CSV: image_id,class_id,confidence,x1,y1,x2,y2. Class IDs
are canonical 0..5 and coordinates are pixels in the original image.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from controlled_common import NAMES, read_json, sha256, write_json


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter) if inter else 0.0


def ground_truth(path: Path):
    doc = read_json(path)
    valid_images = {i["id"] for i in doc["images"]}
    gt = defaultdict(list)
    for a in doc["annotations"]:
        x, y, w, h = a["bbox"]
        gt[(a["image_id"], a["category_id"] - 1)].append((x, y, x + w, y + h))
    return gt, valid_images


def predictions(path: Path, valid_images: set[int]):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = ["image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
        if reader.fieldnames != required:
            raise ValueError(f"prediction header must be exactly {','.join(required)}")
        for n, row in enumerate(reader, 2):
            image_id, class_id = int(row["image_id"]), int(row["class_id"])
            confidence = float(row["confidence"])
            box = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            if image_id not in valid_images or class_id not in range(len(NAMES)):
                raise ValueError(f"unknown image/class at row {n}")
            if not 0 <= confidence <= 1 or not (box[2] > box[0] and box[3] > box[1]):
                raise ValueError(f"invalid score/box at row {n}")
            rows.append((confidence, image_id, class_id, box))
    return sorted(rows, reverse=True)


def class_curve(gt, pred, class_id, threshold):
    relevant = [(s, image, box) for s, image, cls, box in pred if cls == class_id]
    count = sum(len(v) for (image, cls), v in gt.items() if cls == class_id)
    matched = defaultdict(set)
    tp, fp = [], []
    for _, image, box in relevant:
        candidates = gt.get((image, class_id), [])
        best_iou, best_index = 0.0, None
        for index, target in enumerate(candidates):
            overlap = iou(box, target)
            if index not in matched[image] and overlap > best_iou:
                best_iou, best_index = overlap, index
        hit = best_index is not None and best_iou >= threshold
        if hit:
            matched[image].add(best_index)
        tp.append(1 if hit else 0); fp.append(0 if hit else 1)
    cum_tp = []; cum_fp = []; t = f = 0
    for hit, miss in zip(tp, fp):
        t += hit; f += miss; cum_tp.append(t); cum_fp.append(f)
    recalls = [x / count for x in cum_tp] if count else []
    precisions = [x / max(1, x + y) for x, y in zip(cum_tp, cum_fp)]
    ap = 0.0
    if count:
        for level in range(101):
            r = level / 100
            ap += max((p for p, rr in zip(precisions, recalls) if rr >= r), default=0) / 101
    return ap, (cum_tp[-1] if cum_tp else 0), (cum_fp[-1] if cum_fp else 0), count


def evaluate(gt_path: Path, pred_path: Path, confidence_threshold=0.25):
    gt, valid = ground_truth(gt_path)
    pred = predictions(pred_path, valid)
    thresholds = [0.50 + i * 0.05 for i in range(10)]
    per_class = []
    for class_id, name in enumerate(NAMES):
        aps = [class_curve(gt, pred, class_id, threshold)[0] for threshold in thresholds]
        selected = [p for p in pred if p[0] >= confidence_threshold]
        _, tp, fp, positives = class_curve(gt, selected, class_id, 0.5)
        per_class.append({
            "class_id": class_id, "class_name": name,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, positives),
            "map50": aps[0], "map50_95": sum(aps) / len(aps), "ground_truth": positives,
        })
    macro = lambda key: sum(c[key] for c in per_class) / len(per_class)
    return {"confidence_threshold": confidence_threshold, "iou_thresholds": thresholds,
            "precision": macro("precision"), "recall": macro("recall"),
            "map50": macro("map50"), "map50_95": macro("map50_95"), "per_class": per_class}


def self_test():
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        gt = {"images": [{"id": 1, "width": 100, "height": 100}],
              "categories": [{"id": i + 1, "name": n} for i, n in enumerate(NAMES)],
              "annotations": [{"id": i + 1, "image_id": 1, "category_id": i + 1,
                               "bbox": [i, i, 10, 10], "area": 100, "iscrowd": 0} for i in range(6)]}
        write_json(temp / "gt.json", gt)
        (temp / "pred.csv").write_text("image_id,class_id,confidence,x1,y1,x2,y2\n" +
            "".join(f"1,{i},0.9,{i},{i},{i+10},{i+10}\n" for i in range(6)))
        result = evaluate(temp / "gt.json", temp / "pred.csv")
        assert all(abs(result[k] - 1) < 1e-12 for k in ("precision", "recall", "map50", "map50_95"))
    print("PASS: common evaluator synthetic oracle (P=R=mAP50=mAP50:95=1.0)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--release", type=Path, help="required, valid freeze manifest when ground truth is test")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not all((args.ground_truth, args.predictions, args.output)):
            parser.error("ground-truth, predictions, and output are required")
        from test_guard import require_non_test
        inferred_split = "test" if args.ground_truth.stem.endswith("_test") else "val"
        require_non_test(inferred_split, args.release)
        receipt = args.release.with_suffix(args.release.suffix + ".used.json") if inferred_split == "test" else None
        if receipt is not None and receipt.exists():
            raise PermissionError(f"test release already consumed: {receipt}")
        result = evaluate(args.ground_truth, args.predictions, args.confidence_threshold)
        write_json(args.output, result)
        if receipt is not None:
            write_json(receipt, {"release_sha256": sha256(args.release),
                                 "predictions_sha256": sha256(args.predictions),
                                 "result_sha256": sha256(args.output)})
        print(json.dumps(result, indent=2))
