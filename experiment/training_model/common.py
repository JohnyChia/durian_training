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




def _apply_efficientnet_imagenet_normalization(model):
    """
    Apply ImageNet normalization only to the Advanced EfficientNet-B0 backbone.

    Ultralytics supplies RGB tensors scaled to [0, 1], while torchvision
    EfficientNet-B0 DEFAULT weights expect:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    The patch is attached only to model.model[0] (TorchVision backbone), so
    Original/Baseline and the YOLO26 neck/head are unaffected.
    """
    import types

    backbone = model.model.model[0]

    if backbone.__class__.__name__ != "TorchVision":
        raise RuntimeError(
            "ImageNet normalization requested, but model layer 0 is "
            f"{backbone.__class__.__name__}, not TorchVision"
        )

    if getattr(backbone, "_imagenet_normalization_enabled", False):
        print("ImageNet normalization already enabled.")
        return

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32,
    ).view(1, 3, 1, 1)

    backbone.register_buffer(
        "_imagenet_mean",
        mean,
        persistent=False,
    )
    backbone.register_buffer(
        "_imagenet_std",
        std,
        persistent=False,
    )

    original_forward = backbone.forward

    def normalized_forward(self, x):
        mean_t = self._imagenet_mean.to(
            device=x.device,
            dtype=x.dtype,
        )
        std_t = self._imagenet_std.to(
            device=x.device,
            dtype=x.dtype,
        )

        x = (x - mean_t) / std_t
        return original_forward(x)

    backbone.forward = types.MethodType(
        normalized_forward,
        backbone,
    )
    backbone._imagenet_normalization_enabled = True

    print("\n===== EFFICIENTNET INPUT NORMALIZATION =====")
    print("Input range       : Ultralytics RGB [0, 1]")
    print("Mean              : [0.485, 0.456, 0.406]")
    print("Std               : [0.229, 0.224, 0.225]")
    print("Backbone layer    : model.model[0]")
    print("NORMALIZATION GATE: PASS")


def _apply_semantic_yolo26_transfer(model, source_weights):
    """
    Transfer semantically corresponding YOLO26 detector weights into the
    EfficientNet-backed Advanced detector.

    EfficientNet remains ImageNet-pretrained.
    COCO class-output tensors are transferred only when shapes are exactly
    compatible; incompatible 80-class outputs remain newly initialized.
    """
    from ultralytics import YOLO

    source = YOLO(str(source_weights)).model
    target = model.model

    semantic_map = (
        (9,  7,  "SPPF"),
        (10, 8,  "C2PSA"),
        (13, 11, "FPN_P4_C3k2"),
        (16, 14, "FPN_P3_C3k2"),
        (17, 15, "PAN_P3_TO_P4_CONV"),
        (19, 17, "PAN_P4_C3k2"),
        (20, 18, "PAN_P4_TO_P5_CONV"),
        (22, 20, "PAN_P5_C3k2"),
    )

    transferred = 0
    eligible = 0

    print("\n===== SEMANTIC YOLO26 PRETRAINED TRANSFER =====")

    for src_idx, dst_idx, label in semantic_map:
        src_module = source.model[src_idx]
        dst_module = target.model[dst_idx]

        src_state = src_module.state_dict()
        dst_state = dst_module.state_dict()

        new_state = {}

        module_total = sum(v.numel() for v in dst_state.values())
        module_transferred = 0
        matched = 0

        eligible += module_total

        for key, dst_tensor in dst_state.items():
            src_tensor = src_state.get(key)

            if (
                src_tensor is not None
                and src_tensor.shape == dst_tensor.shape
            ):
                new_state[key] = src_tensor
                module_transferred += dst_tensor.numel()
                matched += 1

        dst_module.load_state_dict(new_state, strict=False)

        transferred += module_transferred

        pct = (
            100.0 * module_transferred / module_total
            if module_total
            else 100.0
        )

        print(
            f"{label:24} "
            f"{matched:3d}/{len(dst_state):3d} tensors "
            f"{module_transferred:9,d}/{module_total:9,d} "
            f"{pct:6.2f}%"
        )

    # Detect: official YOLO26 layer 23 -> Advanced layer 21.
    src_detect = source.model[23]
    dst_detect = target.model[21]

    print("\n===== DETECT TRANSFER =====")

    for branch in ("cv2", "one2one_cv2", "cv3", "one2one_cv3"):
        src_branch = getattr(src_detect, branch)
        dst_branch = getattr(dst_detect, branch)

        src_state = src_branch.state_dict()
        dst_state = dst_branch.state_dict()

        new_state = {}

        branch_total = sum(v.numel() for v in dst_state.values())
        branch_transferred = 0
        matched = 0

        eligible += branch_total

        for key, dst_tensor in dst_state.items():
            src_tensor = src_state.get(key)

            if (
                src_tensor is not None
                and src_tensor.shape == dst_tensor.shape
            ):
                new_state[key] = src_tensor
                branch_transferred += dst_tensor.numel()
                matched += 1

        dst_branch.load_state_dict(new_state, strict=False)

        transferred += branch_transferred

        pct = (
            100.0 * branch_transferred / branch_total
            if branch_total
            else 0.0
        )

        print(
            f"{branch:18} "
            f"{matched:3d}/{len(dst_state):3d} tensors "
            f"{branch_transferred:9,d}/{branch_total:9,d} "
            f"{pct:6.2f}%"
        )

    # Full YOLO-side state includes the three new EfficientNet adapters.
    yolo_side_total = sum(
        v.numel()
        for idx in range(4, len(target.model))
        for v in target.model[idx].state_dict().values()
    )

    coverage = (
        100.0 * transferred / yolo_side_total
        if yolo_side_total
        else 0.0
    )

    print("\n===== TRANSFER SUMMARY =====")
    print(f"Transferred state elements : {transferred:,}")
    print(f"YOLO-side state elements   : {yolo_side_total:,}")
    print(f"Semantic coverage          : {coverage:.2f}%")

    if coverage < 60.0:
        raise RuntimeError(
            f"Semantic YOLO26 transfer coverage too low: {coverage:.2f}%"
        )

    print("TRANSFER GATE: PASS")

    del source

    return {
        "transferred": transferred,
        "yolo_side_total": yolo_side_total,
        "coverage": coverage,
    }

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

    if config.get("imagenet_normalize", False):
        _apply_efficientnet_imagenet_normalization(model)

    pretrained_detector = config.get("pretrained_detector")
    if pretrained_detector:
        _apply_semantic_yolo26_transfer(
            model,
            pretrained_detector,
        )

    train_args = {
        "data": str(data.resolve()),
        "epochs": int(train["epochs"]),
        "imgsz": int(train["imgsz"]),
        "batch": train["batch"],
        "device": str(train["device"]),
        "workers": int(train["workers"]),
        "patience": int(train["patience"]),
        "close_mosaic": int(train["close_mosaic"]),
        "project": str(Path(config["project"]).resolve()),
        "name": config["run_name"],
        "seed": int(train["seed"]),
        "deterministic": True,
    }

    optional_train_args = (
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "cos_lr",
        "freeze",
        "mosaic",
        "mixup",
        "scale",
        "translate",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "erasing",
        "amp",
    )

    for key in optional_train_args:
        if key in train:
            train_args[key] = train[key]

    model.train(**train_args)
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
