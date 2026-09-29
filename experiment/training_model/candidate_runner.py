#!/usr/bin/env python3
"""Validated training entry point. It never exposes test to a trainer."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import tempfile
from pathlib import Path

from controlled_common import CONTROL, DATASET, NAMES, read_json, sha256
from test_guard import require_non_test


REQUIRED_TRAINING = {"image_size", "epochs", "batch_size", "optimizer", "learning_rate", "scheduler", "seeds", "early_stopping", "augmentation"}


def materialize_yolo_dataset(destination: Path) -> Path:
    """Write a CWD-independent runtime view of the canonical YOLO YAML."""
    source = DATASET / "dataset.yaml"
    lines = source.read_text(encoding="utf-8").splitlines()
    path_rows = [index for index, line in enumerate(lines) if line.startswith("path:")]
    if path_rows != [0]:
        raise ValueError(f"expected one leading path entry in {source}")
    lines[0] = f"path: {json.dumps(str(DATASET))}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def materialize_dfine_config(destination: Path) -> Path:
    """Resolve path tokens in the frozen D-FINE template at execution time."""
    source = CONTROL / "native" / "dfine_n_custom.yml"
    replacements = {
        "__DURIAN_OUTPUT_DIR__": CONTROL / "outputs" / "dfine_n",
        "__DURIAN_TRAIN_IMAGES__": DATASET / "images" / "train",
        "__DURIAN_VAL_IMAGES__": DATASET / "images" / "val",
        "__DURIAN_TRAIN_ANNOTATIONS__": CONTROL / "coco" / "annotations" / "instances_train.json",
        "__DURIAN_VAL_ANNOTATIONS__": CONTROL / "coco" / "annotations" / "instances_val.json",
    }
    rendered = source.read_text(encoding="utf-8")
    for token, path in replacements.items():
        if rendered.count(token) != 1:
            raise ValueError(f"expected exactly one {token} in {source}")
        rendered = rendered.replace(token, json.dumps(str(path)))
    if "__DURIAN_" in rendered:
        raise ValueError(f"unresolved D-FINE runtime path token in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return destination


def validate(path: Path) -> dict:
    c = read_json(path)
    assert c["candidate"] == path.stem
    assert c["data"]["train"] == "train" and c["data"]["validation"] == "val"
    assert c["data"]["test"] == {"split": "test", "access": "release_manifest_only"}
    assert c["data"]["classes"] == 6 and c["data"]["class_names"] == NAMES
    assert c["selection"]["split"] == "val" and c["selection"]["metric"] == "map50_95"
    assert REQUIRED_TRAINING <= c["training"].keys()
    assert c["training"]["optimizer"] == "AdamW" and c["training"]["seeds"] == [20260917, 20260923, 20261001]
    assert "test" not in json.dumps({"train": c["data"]["train"], "validation": c["data"]["validation"], "selection": c["selection"]})
    return c


def checkpoint(c: dict, weight_dir: Path) -> Path:
    lock = read_json(CONTROL / "locks" / "pretrained_weights.json")
    lock_id = c["initialization"].get("lock_id") or c["initialization"].get("source_lock_id")
    item = lock[lock_id]
    path = weight_dir / item["file"]
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise FileNotFoundError(f"missing/hash-mismatched locked checkpoint: {path}")
    return path


def execute(c: dict, seed: int, weight_dir: Path, dfine_repo: Path | None) -> None:
    require_non_test("train")
    if seed not in c["training"]["seeds"]: raise ValueError("seed is not frozen")
    candidate, t = c["candidate"], c["training"]
    if candidate.startswith("yolo"):
        from ultralytics import YOLO
        init = checkpoint(c, weight_dir) if candidate == "yolo26n" else weight_dir / "yolo26n_p2_semantic_init.pt"
        if not init.is_file(): raise FileNotFoundError(f"generate semantic P2 initialization first: {init}")
        with tempfile.TemporaryDirectory(prefix="durian_yolo_dataset_") as temp:
            data = materialize_yolo_dataset(Path(temp) / "dataset.yaml")
            YOLO(str(init)).train(data=str(data), imgsz=t["image_size"], epochs=t["epochs"],
                batch=t["batch_size"], nbs=t["nominal_batch_size"], optimizer=t["optimizer"], lr0=t["learning_rate"],
                lrf=t["final_lr_fraction"], weight_decay=t["weight_decay"], cos_lr=True, warmup_epochs=t["warmup_epochs"],
                patience=t["early_stopping"]["patience_epochs"], seed=seed, deterministic=True, amp=t["amp"], workers=t["workers"],
                project=str(CONTROL / "outputs" / candidate), name=f"seed_{seed}", exist_ok=False, **t["augmentation"])
    elif candidate == "rfdetr_n":
        from rfdetr import RFDETRNano
        model = RFDETRNano(pretrain_weights=str(checkpoint(c, weight_dir)), resolution=t["image_size"])
        model.train(dataset_dir=str(CONTROL / "coco" / "rfdetr"), epochs=t["epochs"], batch_size=t["batch_size"],
            grad_accum_steps=t["gradient_accumulation_steps"], lr=t["learning_rate"], lr_encoder=t["backbone_learning_rate"],
            weight_decay=t["weight_decay"], lr_drop=t["lr_drop_epoch"], seed=seed, early_stopping=True,
            early_stopping_patience=t["early_stopping"]["patience_epochs"], output_dir=str(CONTROL / "outputs" / candidate / f"seed_{seed}"))
    else:
        if dfine_repo is None: raise ValueError("--dfine-repo is required")
        if subprocess.run(["git", "-C", str(dfine_repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() != c["framework"]["commit"]:
            raise ValueError("D-FINE checkout is not at locked commit")
        native = dfine_repo / "configs" / "dfine" / "controlled_durian_n.yml"
        materialize_dfine_config(native)
        cmd = ["python", "train.py", "-c", str(native), "--use-amp", f"--seed={seed}", "-t", str(checkpoint(c, weight_dir))]
        subprocess.run(cmd, cwd=dfine_repo, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True, type=Path); p.add_argument("--check", action="store_true")
    p.add_argument("--execute", action="store_true"); p.add_argument("--seed", type=int); p.add_argument("--weight-dir", type=Path, default=CONTROL / "weights"); p.add_argument("--dfine-repo", type=Path)
    a = p.parse_args(); config = validate(a.config)
    if a.check: print(f"PASS: {config['candidate']} executable config validated; test is not a trainer input")
    elif a.execute:
        if a.seed is None: p.error("--seed required with --execute")
        execute(config, a.seed, a.weight_dir, a.dfine_repo)
    else: p.error("choose --check or --execute")
