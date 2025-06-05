from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import csv
import shutil

# === 設定區 ===
character_img_path = r"test\predict\128464957_p0_master1200.jpg"
image_folder = r"D:\temp\image"
output_csv = "test/predict/results.csv"
threshold = 0.8

save_has_character = "test/predict/result/has_character"
save_no_character = "test/predict/result/no_character"

# 建立輸出資料夾
os.makedirs(save_has_character, exist_ok=True)
os.makedirs(save_no_character, exist_ok=True)

# 載入模型與 processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 讀取角色圖片
character_image = Image.open(character_img_path)

# 開 CSV 檔案做紀錄
with open(output_csv, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "similarity", "has_character"])

    # 走訪資料夾內所有圖片
    for file_name in os.listdir(image_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, file_name)
            target_image = Image.open(image_path)

            # CLIP 處理：抽取圖片特徵
            inputs = processor(images=[character_image, target_image], return_tensors="pt", padding=True)
            outputs = model.get_image_features(**inputs)

            # 計算 cosine 相似度
            image_features = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            similarity = torch.cosine_similarity(image_features[0], image_features[1], dim=0)
            sim_score = similarity.item()

            has_character = sim_score >= threshold

            # 複製圖片到對應資料夾
            target_folder = save_has_character if has_character else save_no_character
            shutil.copy(image_path, os.path.join(target_folder, file_name))

            # 寫入 CSV
            writer.writerow([file_name, round(sim_score, 4), int(has_character)])

            # 顯示結果
            print(f"{file_name} 相似度: {sim_score:.3f} → {'✅ 有角色' if has_character else '❌ 無角色'}")
