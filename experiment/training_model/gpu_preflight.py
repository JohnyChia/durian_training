#!/usr/bin/env python3
"""Truthful hardware/framework preflight; never fabricates unavailable CUDA."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from pathlib import Path


def inspect():
    result = {"python": platform.python_version(), "frameworks": {
        name: importlib.util.find_spec(name) is not None for name in ("torch", "ultralytics", "rfdetr")
    }}
    try:
        import torch
        result.update({"torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
                       "cuda_available": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count()})
        result["gpus"] = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                result["gpus"].append({"index": i, "name": prop.name, "vram_bytes": prop.total_memory,
                                       "compute_capability": f"{prop.major}.{prop.minor}"})
    except Exception as exc:
        result.update({"cuda_available": False, "error": f"{type(exc).__name__}: {exc}"})
    result["later_measurement_protocol"] = {
        "safe_batch": "binary search per candidate/resolution after dependency review; stop before OOM",
        "training_vram": "torch.cuda.max_memory_allocated after warmup and one optimizer step",
        "inference_vram": "reset peak stats, warm up 50, measure 200 images",
        "latency": "batch=1, identical device/precision, synchronize CUDA, report median/p95 over 200 images",
    }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True, type=Path); a = p.parse_args()
    value = inspect(); a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2))
