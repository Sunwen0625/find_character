# predict_and_sort.py
import os
import shutil
from fastai.vision.all import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="load_learner.*")
#21

def recognition_character(image_dir:str,classification_path:str,learn: Learner) -> None:
    has_dir =  classification_path+"/has_character"
    no_dir =  classification_path+"no_character"
    os.makedirs(has_dir, exist_ok=True)
    os.makedirs(no_dir, exist_ok=True)

    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, fname)
            img = PILImage.create(img_path)
            pred_class, _, _ = learn.predict(img)

            target = has_dir if str(pred_class) == 'has_character' else no_dir
            shutil.copy(img_path, os.path.join(target, fname))

    # 🔁 清空舊資料夾，方便下一輪再丟新圖片
    shutil.rmtree(image_dir)
    os.makedirs(image_dir, exist_ok=True)
    print("✅ 圖片已分類完畢，請至 predicted/ 資料夾進行人工驗證。")

if __name__ == '__main__':
    image_dir = r"C:\Users\user\Downloads\pic" 
    learn = load_learner("test/predict/model.pkl")  
    recognition_character(image_dir, learn)  # 開始辨識圖片
