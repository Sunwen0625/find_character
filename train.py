# train.py
from fastai.vision.all import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="load_learner.*")

def dataset(path:str) -> DataLoaders:
    """
    讀取資料集
    :param path: 資料集路徑
    :return: DataLoaders 物件
    """
    return ImageDataLoaders.from_folder(
        path,
        valid_pct=0.2,
        seed=42,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(),
        num_workers=0,
    )

def model_training(dls:DataLoaders, model_path:str) -> None:
    """
    若已有模型 → 繼續訓練，否則新建一個
    :param dls: DataLoaders 物件
    :param model_path: 模型儲存絕對路徑
    """
    if os.path.exists(model_path):
        learn = load_learner(model_path)
        learn.dls = dls
        print("✅ 開始增強訓練模型...")
        learn.fine_tune(3)
        
    else:
        print("✅ 開始訓練新模型...")
        learn = vision_learner(dls, resnet34, metrics=accuracy)
        learn.model = learn.model.to(torch.device('cuda:0'))  # 將模型移到 GPU 上
        learn.fine_tune(10)

    learn.export(model_path)
    print("✅ 訓練完成，模型已更新為 model.pkl")

# ====== 用人工驗證過的資料重新訓練 ======
if __name__ == '__main__':
    
    dls = dataset(r"test\predict\augment_positive_images",)  # 讀取資料集
    
    model_path = r"D:\program\py\illegal_stop\test\predict\model.pkl"
    
    model_training(dls, model_path)  # 訓練模型

