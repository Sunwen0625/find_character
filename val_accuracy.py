"""Small manual prediction helper."""

from fastai.vision.all import PILImage, load_learner


if __name__ == "__main__":
    learn = load_learner("test/predict/result/model.pkl")
    img = PILImage.create("test/predict/result/has_character/example.jpg")
    pred_class, pred_idx, probs = learn.predict(img)

    print(f"預測類別：{pred_class}")
    print(f"機率分布：{probs}")
