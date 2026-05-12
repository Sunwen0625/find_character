"""Compatibility exports for prediction and sorting helpers."""

from fastai.vision.all import load_learner

from find_character.predict import recognition_character


if __name__ == "__main__":
    image_dir = "test/predict/input"
    classification_path = "test/predict/predicted"
    learn = load_learner("test/predict/model.pkl")
    recognition_character(image_dir, classification_path, learn)
