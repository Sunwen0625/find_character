"""CLIP-based image similarity sorting utility."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def sort_by_clip_similarity(
    character_img_path: str | Path,
    image_folder: str | Path,
    output_csv: str | Path,
    save_has_character: str | Path,
    save_no_character: str | Path,
    threshold: float = 0.8,
    model_name: str = "openai/clip-vit-base-patch32",
) -> None:
    """Sort images by cosine similarity to a reference character image."""

    source = Path(image_folder)
    has_dir = Path(save_has_character)
    no_dir = Path(save_no_character)
    csv_path = Path(output_csv)
    has_dir.mkdir(parents=True, exist_ok=True)
    no_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    character_image = Image.open(character_img_path)

    with csv_path.open(mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "similarity", "has_character"])

        for image_path in source.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            target_image = Image.open(image_path)
            inputs = processor(images=[character_image, target_image], return_tensors="pt", padding=True)
            outputs = model.get_image_features(**inputs)
            image_features = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            similarity = torch.cosine_similarity(image_features[0], image_features[1], dim=0)
            sim_score = similarity.item()
            has_character = sim_score >= threshold

            target_folder = has_dir if has_character else no_dir
            shutil.copy2(image_path, target_folder / image_path.name)
            writer.writerow([image_path.name, round(sim_score, 4), int(has_character)])
            print(f"{image_path.name} 相似度: {sim_score:.3f} → {'✅ 有角色' if has_character else '❌ 無角色'}")
