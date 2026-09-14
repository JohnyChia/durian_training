#!/usr/bin/env python3
"""Shared, config-driven helpers for the three training experiments."""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


WORKSPACE = Path("/home/johny/durian_ws")
EXPECTED_DISEASE_CLASSES = [
    "leaf_algal",
    "leaf_blight",
    "leaf_colletotrichum",
    "leaf_healthy",
    "leaf_phomopsis",
    "leaf_rhizoctonia",
]


def load_config(path: Path) -> dict:
    path = path.resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    config["_config_path"] = str(path)
    return config


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_device(device: str):
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run where nvidia-smi works, or explicitly set device: cpu "
            "in the experiment YAML if slow CPU training is intentional."
        )


def validate_detection_yaml(path: Path, expected_names: list[str]) -> dict:
    checked = check_det_dataset(str(path.resolve()))
    names = [checked["names"][index] for index in sorted(checked["names"])]
    if names != expected_names:
        raise ValueError(f"Unexpected class order in {path}: {names}; expected {expected_names}")
    for split in ("train", "val", "test"):
        if not checked.get(split):
            raise ValueError(f"Missing {split} path in {path}")
    return checked


def prepare_output(path: Path, overwrite: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite trained model {path}; pass --overwrite explicitly")


def atomic_copy(source: Path, destination: Path):
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def json_metrics(metrics) -> dict:
    return {key: float(value) for key, value in metrics.results_dict.items()}


def train_yolo_experiment(config: dict, overwrite: bool, check_only: bool):
    output = Path(config["output"])
    data = Path(config["data"])
    expected = config["expected_classes"]
    checked = validate_detection_yaml(data, expected)
    summary = {
        "experiment": config["experiment"],
        "config": config["_config_path"],
        "model": config["model"],
        "data": str(data.resolve()),
        "classes": expected,
        "train": checked["train"],
        "val": checked["val"],
        "test": checked["test"],
        "output": str(output.resolve()),
    }
    if check_only:
        model_path = Path(config["model"])
        if model_path.suffix.lower() in {".yaml", ".yml"} and model_path.is_file():
            architecture = YOLO(str(model_path.resolve()))
            if len(architecture.names) != len(expected):
                raise ValueError(
                    f"Architecture output count {len(architecture.names)} does not match "
                    f"dataset class count {len(expected)}"
                )
            summary["architecture"] = {
                "task": architecture.task,
                "parameters": sum(parameter.numel() for parameter in architecture.model.parameters()),
                "stride": architecture.model.stride.tolist(),
            }
        print(json.dumps({"check_only": True, **summary}, indent=2))
        return

    prepare_output(output, overwrite)
    train = config["train"]
    require_device(str(train["device"]))
    seed_everything(int(train["seed"]))
    model = YOLO(config["model"])
    model.train(
        data=str(data.resolve()),
        epochs=int(train["epochs"]),
        imgsz=int(train["imgsz"]),
        batch=train["batch"],
        device=str(train["device"]),
        workers=int(train["workers"]),
        patience=int(train["patience"]),
        close_mosaic=int(train["close_mosaic"]),
        project=str(Path(config["project"]).resolve()),
        name=config["run_name"],
        seed=int(train["seed"]),
        deterministic=True,
    )
    best = Path(model.trainer.best)
    if not best.is_file():
        raise FileNotFoundError(f"Ultralytics did not produce best weights: {best}")
    atomic_copy(best, output)

    final_model = YOLO(output)
    evaluations = {}
    for name, evaluation_yaml in config["evaluations"].items():
        evaluation_yaml = Path(evaluation_yaml)
        validate_detection_yaml(evaluation_yaml, expected)
        metrics = final_model.val(
            data=str(evaluation_yaml.resolve()),
            split="test",
            imgsz=int(train["imgsz"]),
            device=str(train["device"]),
            workers=int(train["workers"]),
        )
        evaluations[name] = json_metrics(metrics)
    report = {**summary, "source_best": str(best.resolve()), "evaluations": evaluations}
    report_path = Path(config["project"]) / config["run_name"] / "final_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
