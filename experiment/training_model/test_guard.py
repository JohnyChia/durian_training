#!/usr/bin/env python3
"""One-way test release control tied to frozen configs and final checkpoints."""
from __future__ import annotations

import argparse
import secrets
from pathlib import Path

from controlled_common import CONTROL, read_json, sha256, write_json


def create_release(checkpoints: list[Path], output: Path) -> None:
    if not checkpoints or any(not p.is_file() for p in checkpoints):
        raise ValueError("all final checkpoints must exist before test release")
    configs = sorted((CONTROL / "configs").glob("*.json"))
    write_json(output, {
        "release_version": 1, "purpose": "single final test evaluation",
        "nonce": secrets.token_hex(16),
        "configs": {p.name: sha256(p) for p in configs},
        "checkpoints": {str(p.resolve()): sha256(p) for p in checkpoints},
        "dataset_freeze": sha256(CONTROL / "coco" / "manifest.json"),
    })


def validate_release(path: Path) -> bool:
    doc = read_json(path)
    if doc.get("purpose") != "single final test evaluation": return False
    if doc.get("dataset_freeze") != sha256(CONTROL / "coco" / "manifest.json"): return False
    if any(sha256(CONTROL / "configs" / name) != digest for name, digest in doc["configs"].items()): return False
    return all(Path(name).is_file() and sha256(Path(name)) == digest for name, digest in doc["checkpoints"].items())


def require_non_test(split: str, release: Path | None = None) -> None:
    if split == "test" and (release is None or not validate_release(release)):
        raise PermissionError("test is sealed; create a valid final release manifest after model/config selection")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze"); freeze.add_argument("--checkpoint", action="append", type=Path, required=True); freeze.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("check"); check.add_argument("--split", choices=["train", "val", "test"], required=True); check.add_argument("--release", type=Path)
    a = p.parse_args()
    if a.command == "freeze": create_release(a.checkpoint, a.output)
    else: require_non_test(a.split, a.release); print(f"PASS: access permitted for {a.split}")
