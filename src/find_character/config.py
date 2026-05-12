"""Configuration loading for the find-character workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WorkflowFlags:
    """Feature switches for each pipeline stage."""

    do_recognition: bool = True
    do_augmentation: bool = False
    do_training: bool = False


@dataclass(frozen=True)
class WorkflowPaths:
    """Filesystem locations used by the workflow."""

    dataset_path: Path
    image_dir: Path
    predicted_folder: Path
    model_path: Path
    verified_folder: Path
    augment_folder: Path
    temp_folder: Path


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    flags: WorkflowFlags
    paths: WorkflowPaths
    n_augments_per_image: int = 3


def _path(value: str | Path) -> Path:
    return Path(value).expanduser()


def load_config(config_path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}

    arg = raw.get("arg", {})
    path_config = raw.get("path", {})

    required_paths = {
        "dataset_path",
        "image_dir",
        "predicted_folder",
        "model_path",
        "verified_folder",
        "augment_folder",
        "temp_folder",
    }
    missing_paths = sorted(required_paths.difference(path_config))
    if missing_paths:
        missing = ", ".join(missing_paths)
        raise ValueError(f"Missing required path config value(s): {missing}")

    return AppConfig(
        flags=WorkflowFlags(
            do_recognition=bool(arg.get("do_recognition", True)),
            do_augmentation=bool(arg.get("do_augmentation", False)),
            do_training=bool(arg.get("do_training", False)),
        ),
        paths=WorkflowPaths(
            dataset_path=_path(path_config["dataset_path"]),
            image_dir=_path(path_config["image_dir"]),
            predicted_folder=_path(path_config["predicted_folder"]),
            model_path=_path(path_config["model_path"]),
            verified_folder=_path(path_config["verified_folder"]),
            augment_folder=_path(path_config["augment_folder"]),
            temp_folder=_path(path_config["temp_folder"]),
        ),
        n_augments_per_image=int(raw.get("n_augments_per_image", 3)),
    )
