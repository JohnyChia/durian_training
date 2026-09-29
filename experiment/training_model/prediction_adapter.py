#!/usr/bin/env python3
"""Normalize Ultralytics JSON or COCO result JSON to common prediction CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def adapt(framework: str, source: Path, output: Path) -> None:
    data = json.loads(source.read_text())
    rows = []
    if framework in {"rfdetr_n", "dfine_n"}:
        for item in data:
            x, y, w, h = item["bbox"]
            rows.append([item["image_id"], item["category_id"] - 1, item["score"], x, y, x + w, y + h])
    else:
        # Export contract: [{image_id, boxes:[{class_id, confidence, xyxy:[...]}]}]
        for image in data:
            for box in image["boxes"]:
                rows.append([image["image_id"], box["class_id"], box["confidence"], *box["xyxy"]])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--framework", required=True, choices=["yolo26n", "yolo26n_p2", "rfdetr_n", "dfine_n"])
    p.add_argument("--input", required=True, type=Path); p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(); adapt(a.framework, a.input, a.output)
