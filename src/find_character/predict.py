"""Image recognition and sorting utilities."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

from fastai.vision.all import Learner, PILImage

warnings.filterwarnings("ignore", category=UserWarning, message="load_learner.*")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def recognition_character(image_dir: str | Path, classification_path: str | Path, learn: Learner) -> None:
    """Classify images into ``has_character`` and ``no_character`` folders."""

    source_dir = Path(image_dir)
    output_dir = Path(classification_path)
    has_dir = output_dir / "has_character"
    no_dir = output_dir / "no_character"
    has_dir.mkdir(parents=True, exist_ok=True)
    no_dir.mkdir(parents=True, exist_ok=True)

    for image_path in source_dir.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        img = PILImage.create(image_path)
        pred_class, _, _ = learn.predict(img)
        target_dir = has_dir if str(pred_class) == "has_character" else no_dir
        shutil.copy2(image_path, target_dir / image_path.name)

    shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    print("✅ 圖片已分類完畢，請至 predicted/ 資料夾進行人工驗證。")
