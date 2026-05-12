"""Command-line entry point for the find-character workflow."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from fastai.vision.all import load_learner

from find_character.augment import process_images
from find_character.config import load_config
from find_character.predict import recognition_character
from find_character.train import create_dataset, model_training


def run(config_path: str | Path) -> None:
    """Run the configured recognition, augmentation, and training workflow."""

    config = load_config(config_path)

    if config.flags.do_recognition:
        learn = load_learner(config.paths.model_path)
        recognition_character(config.paths.image_dir, config.paths.predicted_folder, learn)

    recognized_has_character = config.paths.predicted_folder / "has_character"
    if config.flags.do_augmentation:
        process_images(
            processed_folder=config.paths.verified_folder,
            unprocessed_folder=recognized_has_character,
            dst_folder=config.paths.augment_folder,
            n_augments_per_image=config.n_augments_per_image,
        )
    elif recognized_has_character.exists():
        config.paths.verified_folder.mkdir(parents=True, exist_ok=True)
        for image_path in recognized_has_character.iterdir():
            if image_path.is_file():
                shutil.move(str(image_path), str(config.paths.verified_folder / image_path.name))

    if config.flags.do_training:
        dls = create_dataset(config.paths.dataset_path)
        model_training(dls, config.paths.model_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Recognize, augment, and train a character image classifier.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML config file. Defaults to configs/config.yaml.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and run the workflow."""

    args = build_parser().parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
