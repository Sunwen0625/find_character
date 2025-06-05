import os
import cv2
import shutil
import albumentations as A
import numpy as np


# ======= 增強操作設定 =======
def get_transform() -> A.Compose:
    return A.Compose([
    # ✅ 結構類（模擬不同角度和構圖）
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Affine(
        scale=(0.9, 1.1),               # 微縮放
        translate_percent=(0, 0.05),    # 小範圍平移
        rotate=(-15, 15),               # 輕微旋轉
        p=0.5,
    ),

    # ✅ 光影類（模擬不同光照條件）
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=10,
        val_shift_limit=10,
        p=0.3
    ),

    # ✅ 模糊/雜訊類（應對低品質圖）
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.MotionBlur(blur_limit=3),
    ], p=0.2),

    # ✅ 模擬拍攝環境（煙霧/陰影）
    A.RandomShadow(p=0.2),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.1),

    # ✅ 提升局部細節（對比強化）
    A.CLAHE(p=0.2),
])

#讀取圖片檔案
def load_image(filepath: str) -> np.ndarray|None: # ✅ 正確支援中文路徑
    try:
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"❌ 無法開啟 {filepath}：{e}")
        return None
        
#增強圖片並保存
def augment_image_and_save(image: np.ndarray, base_name: str, dst_folder: str, transform: A.Compose, N_AUGMENTS_PER_IMAGE: int) -> tuple:
    count_success, count_fail = 0, 0
    for i in range(N_AUGMENTS_PER_IMAGE):
        try:
            augmented = transform(image=image)['image']
            new_name = f"{base_name}_aug_{i}.jpg"
            save_path = os.path.join(dst_folder, new_name)

            # ✅ 正確寫入含中文檔名
            success, encoded_image = cv2.imencode('.jpg', augmented)
            if success:
                with open(save_path, mode='wb') as f:
                    encoded_image.tofile(f)
                print(f"✅ 生成圖片：{new_name}")
                count_success += 1
            else:
                print(f"❌ 編碼失敗：{new_name}")
                count_fail += 1

        except Exception as e:
            print(f"❌ 寫入失敗 {new_name}：{e}")
            count_fail += 1
    return count_success, count_fail

#移動原始圖片
def move_original_image(filepath: str, processed_folder: str) -> None:
    try:
        dst_path = os.path.join(processed_folder, os.path.basename(filepath))
        shutil.move(filepath, dst_path)
        print(f"📦 移動原圖至 verified：{os.path.basename(filepath)}")
    except Exception as e:
        print(f"❌ 原圖移動失敗：{os.path.basename(filepath)} → {e}")

#主程序
def process_images(processed_folder: str, unprocessed_folder: str, dst_folder: str, N_AUGMENTS_PER_IMAGE: int) -> None:
    os.makedirs(dst_folder, exist_ok=True)
    transform = get_transform()

    count_total_success, count_total_fail = 0, 0
    for fname in os.listdir(unprocessed_folder):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):continue

        filepath = os.path.join(unprocessed_folder, fname)
        image = load_image(filepath)

        if image is None:
            print(f"❌ 讀不到圖片內容（可能壞圖）：{fname}")
            count_fail += 1
            continue

        base_name = os.path.splitext(fname)[0]
        count_success, count_fail = augment_image_and_save(
            image, 
            base_name, 
            dst_folder, 
            transform, 
            N_AUGMENTS_PER_IMAGE
            )
        count_total_success += count_success
        count_total_fail += count_fail
        move_original_image(filepath, processed_folder)
    print(f"\n🚀 增強完畢：成功 {count_total_success} 張，失敗 {count_total_fail} 張")


if __name__ == '__main__':
    # ======= 設定 =======
    processed_folder = r"test\predict\verified\has_character"  # ✅ 要新增的 Verified 路徑
    unprocessed_folder = r"test\predict\temp_correct_image"  # 人工驗證過的資料夾
    dst_folder = r"test\predict\augment_positive_images\has_character"  # 增強圖輸出位置
    N_AUGMENTS_PER_IMAGE = 2  # 每張原圖要產生幾張增強圖    

    process_images(
        processed_folder,
        unprocessed_folder, 
        dst_folder, 
        N_AUGMENTS_PER_IMAGE
        )