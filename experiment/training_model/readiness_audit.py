#!/usr/bin/env python3
"""Single, behavioral readiness gate for controlled detector training."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from candidate_runner import FROZEN_SEEDS, materialize_dfine_config, materialize_yolo_dataset, validate as validate_config
from common_evaluator import self_test as evaluator_self_test
from controlled_common import CONTROL, DATASET, NAMES, dataset_records, read_json, sha256
from prediction_adapter import adapt
from semantic_transfer_yolo26_p2 import verify_plan
from test_guard import create_release, require_non_test, validate_release
from verify_coco_view import verify as verify_coco


def tree_hash(kind: str) -> str:
    h = hashlib.sha256()
    for path in sorted((DATASET / kind).glob("*/*")):
        h.update(path.relative_to(DATASET).as_posix().encode() + b"\0")
        h.update(bytes.fromhex(sha256(path)))
    return h.hexdigest()


def check_dataset():
    lock = read_json(CONTROL / "locks" / "control_lock.json")["canonical_dataset"]
    assert sha256(DATASET / "manifests" / "audit.json") == lock["audit_json_sha256"]
    assert sha256(DATASET / "manifests" / "dataset_manifest.json") == lock["dataset_manifest_sha256"]
    assert tree_hash("images") == lock["images_tree_sha256"] and tree_hash("labels") == lock["labels_tree_sha256"]
    records = dataset_records()
    assert {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")} == {"train": 144, "val": 62, "test": 46}
    assert sum(len(r["boxes"]) for r in records) == 5886


def check_configs():
    lock = read_json(CONTROL / "locks" / "control_lock.json")
    configs = {}
    for name, digest in lock["configs"].items():
        path = CONTROL / "configs" / name
        assert sha256(path) == digest
        configs[path.stem] = validate_config(path)
    for name, digest in lock["native_configs"].items(): assert sha256(CONTROL / "native" / name) == digest
    for name, digest in lock["code"].items(): assert sha256(Path(__file__).parent / name) == digest
    dfine_native = (CONTROL / "native" / "dfine_n_custom.yml").read_text()
    assert "images/test" not in dfine_native and "instances_test" not in dfine_native
    assert "remap_mscoco_category: true" in dfine_native
    assert "epochs: 100" in dfine_native
    assert "epoch: 88" in dfine_native and "stop_epoch: 88" in dfine_native
    assert len(configs) == 4
    assert all(c["training"]["epochs"] == 100 for c in configs.values())
    assert all(c["training"]["seeds"] == FROZEN_SEEDS for c in configs.values())
    assert all(c["training"]["early_stopping"]["enabled"] is False for c in configs.values())
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        yolo_runtime = materialize_yolo_dataset(temp / "dataset.yaml").read_text()
        assert "\ntest:" not in f"\n{yolo_runtime}"
        dfine_output = temp / "outputs" / "dfine_n" / f"seed_{FROZEN_SEEDS[0]}"
        dfine_runtime = materialize_dfine_config(temp / "dfine.yml", dfine_output).read_text()
        assert str(dfine_output) in dfine_runtime
        assert "images/test" not in dfine_runtime and "instances_test" not in dfine_runtime


def check_test_guard():
    try: require_non_test("test")
    except PermissionError: pass
    else: raise AssertionError("test access was not denied")
    require_non_test("train"); require_non_test("val")
    assert {p.name for p in (CONTROL / "coco" / "rfdetr").iterdir()} == {"train", "valid"}
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        checkpoint = temp / "final.pt"; checkpoint.write_bytes(b"synthetic-final-checkpoint")
        release = temp / "release.json"; create_release([checkpoint], release)
        assert validate_release(release); require_non_test("test", release)
        pred = temp / "empty.csv"
        pred.write_text("image_id,class_id,confidence,x1,y1,x2,y2\n")
        evaluator = str(Path(__file__).parent / "common_evaluator.py")
        synthetic_gt = temp / "instances_test.json"
        synthetic_gt.write_text(json.dumps({"images": [{"id": 1, "width": 10, "height": 10}], "annotations": [],
                                             "categories": [{"id": i + 1, "name": n} for i, n in enumerate(NAMES)]}))
        allowed = [sys.executable, evaluator, "--ground-truth", str(synthetic_gt), "--predictions", str(pred),
                   "--output", str(temp / "synthetic-result.json"), "--release", str(release)]
        first = subprocess.run(allowed, capture_output=True, text=True)
        assert first.returncode == 0 and release.with_suffix(".json.used.json").is_file()
        repeated = subprocess.run(allowed, capture_output=True, text=True)
        assert repeated.returncode != 0 and "already consumed" in repeated.stderr
        command = [sys.executable, evaluator, "--ground-truth",
                   str(CONTROL / "coco" / "annotations" / "instances_test.json"), "--predictions", str(pred), "--output", str(temp / "out.json")]
        denied = subprocess.run(command, capture_output=True, text=True)
        assert denied.returncode != 0 and "test is sealed" in denied.stderr


def check_evaluator_and_adapters():
    evaluator_self_test()
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp); src = temp / "coco.json"; out = temp / "out.csv"
        src.write_text('[{"image_id":1,"category_id":2,"score":0.75,"bbox":[1,2,3,4]}]')
        adapt("rfdetr_n", src, out)
        rows = list(csv.DictReader(out.open()))
        assert rows == [{"image_id": "1", "class_id": "1", "confidence": "0.75", "x1": "1", "y1": "2", "x2": "4", "y2": "6"}]


def check_dependencies_and_weights():
    frameworks = read_json(CONTROL / "locks" / "frameworks.json")
    assert frameworks["ultralytics"]["package"] == "ultralytics==8.4.152"
    assert frameworks["rfdetr"]["package"] == "rfdetr[train]==1.11.0"
    assert len(frameworks["dfine"]["commit"]) == 40 and frameworks["base"]["torch"] == "2.6.0+cu124"
    for req in (CONTROL / "locks").glob("requirements-*.txt"):
        for line in req.read_text().splitlines():
            if line and not line.startswith(("#", "--")): assert "==" in line, f"unlocked dependency: {line}"
    weights = read_json(CONTROL / "locks" / "pretrained_weights.json")
    assert set(weights) == {"yolo26n_coco", "rfdetr_nano_coco", "dfine_n_coco"}
    for value in weights.values(): assert value["source"].startswith("https://") and len(value["sha256"]) == 64 and value["bytes"] > 0


def check_p2():
    verify_plan()
    lock = read_json(CONTROL / "locks" / "control_lock.json")["generated_evidence"]
    path = CONTROL / "reports" / "p2_transfer_verification.json"
    assert sha256(path) == lock["p2_transfer_verification.json"]
    report = read_json(path)
    assert report["forward_640_verified"] and report["copied_tensors"] == report["source_tensors"]
    assert report["copied_elements"] / report["target_elements"] > 0.95 and not report["skipped"]


def check_gpu():
    report = read_json(CONTROL / "reports" / "gpu_preflight.json")
    assert {"python", "frameworks", "cuda_available", "gpu_count", "gpus", "later_measurement_protocol"} <= report.keys()
    assert report["gpu_count"] == len(report["gpus"])
    if not report["cuda_available"]: assert report["gpu_count"] == 0 and report["gpus"] == []


def main() -> int:
    checks = [
        ("Dataset identity frozen", check_dataset),
        ("Train/Val/Test split frozen", lambda: (check_dataset(), verify_coco())),
        ("Test access protected", check_test_guard),
        ("YOLO26n config frozen", lambda: validate_config(CONTROL / "configs" / "yolo26n.json")),
        ("YOLO26n-P2 config frozen", lambda: (validate_config(CONTROL / "configs" / "yolo26n_p2.json"), check_p2())),
        ("RF-DETR-N config frozen", lambda: validate_config(CONTROL / "configs" / "rfdetr_n.json")),
        ("D-FINE-N config frozen", lambda: validate_config(CONTROL / "configs" / "dfine_n.json")),
        ("YOLO <-> COCO identity verified", verify_coco),
        ("Common evaluator verified", check_evaluator_and_adapters),
        ("Dependency versions locked", lambda: (check_configs(), check_dependencies_and_weights())),
        ("Pretrained weights/hashes locked", check_dependencies_and_weights),
        ("GPU preflight completed", check_gpu),
    ]
    failed = []
    for label, function in checks:
        try: function(); print(f"[PASS] {label}")
        except Exception as exc:
            failed.append((label, exc)); print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
    print()
    if failed:
        print("NO-GO — FIXES REQUIRED"); return 1
    print("PASS — CONTROLLED TRAINING READY"); return 0


if __name__ == "__main__": raise SystemExit(main())
