"""Training helpers for the character classifier."""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from fastai.vision.all import DataLoaders, ImageDataLoaders, Resize, accuracy, aug_transforms, load_learner, resnet34, vision_learner

warnings.filterwarnings("ignore", category=UserWarning, message="load_learner.*")


def create_dataset(path: str | Path) -> DataLoaders:
    """Create FastAI dataloaders from a folder-based image dataset."""

    return ImageDataLoaders.from_folder(
        Path(path),
        valid_pct=0.2,
        seed=42,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(),
        num_workers=0,
    )


def model_training(dls: DataLoaders, model_path: str | Path, epochs_existing: int = 3, epochs_new: int = 10) -> None:
    """Continue training an existing model or train a new ResNet-34 classifier."""

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        learn = load_learner(destination)
        learn.dls = dls
        print("✅ 開始增強訓練模型...")
        learn.fine_tune(epochs_existing)
    else:
        print("✅ 開始訓練新模型...")
        learn = vision_learner(dls, resnet34, metrics=accuracy)
        if torch.cuda.is_available():
            learn.model = learn.model.to(torch.device("cuda:0"))
        learn.fine_tune(epochs_new)

    learn.export(destination)
    print(f"✅ 訓練完成，模型已更新為 {destination}")
