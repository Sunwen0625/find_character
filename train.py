"""Compatibility exports for training helpers."""

from find_character.train import create_dataset, model_training

# Backward-compatible name used by older scripts.
dataset = create_dataset


if __name__ == "__main__":
    dls = create_dataset("test/predict/augment_positive_images")
    model_training(dls, "test/predict/model.pkl")
