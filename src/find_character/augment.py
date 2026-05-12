"""Image augmentation utilities."""

from __future__ import annotations

import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_transform() -> A.Compose:
    """Build the augmentation pipeline used for positive character images."""

    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0, 0.05), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=3)], p=0.2),
            A.RandomShadow(p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.1),
            A.CLAHE(p=0.2),
        ]
    )


def load_image(filepath: str | Path) -> np.ndarray | None:
    """Load an image from disk, including paths with non-ASCII characters."""

    path = Path(filepath)
    try:
        image_bytes = path.read_bytes()
        image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except OSError as error:
        print(f"❌ 無法開啟 {path}：{error}")
        return None


def augment_image_and_save(
    image: np.ndarray,
    base_name: str,
    dst_folder: str | Path,
    transform: A.Compose,
    n_augments_per_image: int,
) -> tuple[int, int]:
    """Generate augmented images and save them as JPEG files."""

    destination = Path(dst_folder)
    count_success, count_fail = 0, 0
    for index in range(n_augments_per_image):
        new_name = f"{base_name}_aug_{index}.jpg"
        save_path = destination / new_name
        try:
            augmented = transform(image=image)["image"]
            success, encoded_image = cv2.imencode(".jpg", augmented)
            if success:
                save_path.write_bytes(encoded_image.tobytes())
                print(f"✅ 生成圖片：{new_name}")
                count_success += 1
            else:
                print(f"❌ 編碼失敗：{new_name}")
                count_fail += 1
        except (cv2.error, OSError, ValueError) as error:
            print(f"❌ 寫入失敗 {new_name}：{error}")
            count_fail += 1

    return count_success, count_fail


def move_original_image(filepath: str | Path, processed_folder: str | Path) -> None:
    """Move the original image into the verified folder after augmentation."""

    source = Path(filepath)
    destination = Path(processed_folder) / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(source), str(destination))
        print(f"📦 移動原圖至 verified：{source.name}")
    except OSError as error:
        print(f"❌ 原圖移動失敗：{source.name} → {error}")


def process_images(
    processed_folder: str | Path,
    unprocessed_folder: str | Path,
    dst_folder: str | Path,
    n_augments_per_image: int,
) -> None:
    """Augment all pending positive images and move originals to verified storage."""

    destination = Path(dst_folder)
    source = Path(unprocessed_folder)
    destination.mkdir(parents=True, exist_ok=True)
    transform = get_transform()

    count_total_success, count_total_fail = 0, 0
    for filepath in source.iterdir():
        if not filepath.is_file() or filepath.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image = load_image(filepath)
        if image is None:
            print(f"❌ 讀不到圖片內容（可能壞圖）：{filepath.name}")
            count_total_fail += 1
            continue

        count_success, count_fail = augment_image_and_save(
            image,
            filepath.stem,
            destination,
            transform,
            n_augments_per_image,
        )
        count_total_success += count_success
        count_total_fail += count_fail
        move_original_image(filepath, processed_folder)

    print(f"\n🚀 增強完畢：成功 {count_total_success} 張，失敗 {count_total_fail} 張")
