#!/usr/bin/env python3
"""Semantically transfer YOLO26n COCO weights into the official P2 graph.

The mapping follows graph roles: identical backbone/top-down P3 nodes, then
P4/P5 PAN nodes shifted by the inserted P2 branch, and detection branches
P3/P4/P5 shifted from indices 0/1/2 to 1/2/3. Shape is verified per tensor.
New P2 nodes and the P2 detection branch remain deterministically initialized.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LAYER_ROLES = {
    **{i: i for i in range(17)},
    17: 23, 18: 24, 19: 25, 20: 26, 21: 27, 22: 28,
}
DETECT_BRANCHES = {0: 1, 1: 2, 2: 3}


def target_key(key: str) -> str | None:
    parts = key.split(".")
    if len(parts) < 3 or parts[0] != "model": return None
    layer = int(parts[1])
    if layer in LAYER_ROLES:
        parts[1] = str(LAYER_ROLES[layer]); return ".".join(parts)
    if layer != 23 or len(parts) < 5: return None
    if parts[2] not in {"cv2", "cv3", "one2one_cv2", "one2one_cv3"}: return None
    branch = int(parts[3])
    if branch not in DETECT_BRANCHES: return None
    parts[1], parts[3] = "29", str(DETECT_BRANCHES[branch])
    return ".".join(parts)


def verify_plan() -> None:
    assert target_key("model.0.conv.weight") == "model.0.conv.weight"
    assert target_key("model.17.conv.weight") == "model.23.conv.weight"
    assert target_key("model.22.cv1.conv.weight") == "model.28.cv1.conv.weight"
    assert target_key("model.23.cv2.0.0.conv.weight") == "model.29.cv2.1.0.conv.weight"
    assert target_key("model.23.one2one_cv3.2.2.bias") == "model.29.one2one_cv3.3.2.bias"
    print("PASS: semantic P2 graph mapping verified (P2/new head branch intentionally excluded)")


def transfer(source: Path, architecture: Path, output: Path, report: Path) -> None:
    import torch
    from ultralytics import YOLO
    source_model = YOLO(str(source)).model.float()
    target_model = YOLO(str(architecture), task="detect").model.float()
    src, dst = source_model.state_dict(), target_model.state_dict()
    copied, skipped = {}, {}
    for source_name, value in src.items():
        destination = target_key(source_name)
        if destination is None: skipped[source_name] = "no semantic counterpart"; continue
        if destination not in dst: skipped[source_name] = "target key absent"; continue
        if tuple(value.shape) != tuple(dst[destination].shape):
            skipped[source_name] = f"shape {tuple(value.shape)} != {tuple(dst[destination].shape)}"; continue
        dst[destination] = value.clone(); copied[source_name] = destination
    target_model.load_state_dict(dst, strict=True)
    target_model.eval()
    with torch.no_grad():
        result = target_model(torch.zeros(1, 3, 640, 640))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": target_model.half(), "semantic_transfer": copied}, output)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "source_tensors": len(src), "target_tensors": len(dst), "copied_tensors": len(copied),
        "copied_elements": sum(src[k].numel() for k in copied),
        "target_elements": sum(v.numel() for v in dst.values()),
        "forward_640_verified": result is not None, "mapping": copied, "skipped": skipped,
    }, indent=2, sort_keys=True) + "\n")
    print(f"PASS: copied {len(copied)} semantic tensors; verified 640x640 forward")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--verify-plan", action="store_true")
    p.add_argument("--source", type=Path); p.add_argument("--architecture", type=Path)
    p.add_argument("--output", type=Path); p.add_argument("--report", type=Path)
    a = p.parse_args()
    if a.verify_plan: verify_plan()
    else:
        if not all((a.source, a.architecture, a.output, a.report)): p.error("source, architecture, output, report required")
        transfer(a.source, a.architecture, a.output, a.report)
