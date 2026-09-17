#!/usr/bin/env python3
"""Train the six-class balanced real/Gazebo baseline model."""

import argparse
from pathlib import Path

from common import load_config, train_yolo_experiment

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "yaml/baseline.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train_yolo_experiment(load_config(args.config), args.overwrite, args.check_only)


if __name__ == "__main__":
    main()
