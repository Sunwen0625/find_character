from fastai.vision.all import *

# 讀入模型
learn = load_learner(r"test\predict\result\model.pkl")  # ← 換成你的檔名

img = PILImage.create(r"test\predict\result\has_character\LINE_ALBUM_要ㄌㄌ自己找39_250418_314.jpg")  # 換成你的圖片路徑
pred_class, pred_idx, probs = learn.predict(img)

print(f"預測類別：{pred_class}")
print(f"機率分布：{probs}")
