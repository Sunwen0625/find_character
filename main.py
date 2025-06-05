import yaml
from fastai.vision.all import *
import warnings

from .predict_and_sort import recognition_character
from .augment_positive_images import process_images
from .train import model_training
warnings.filterwarnings("ignore", category=UserWarning, message="load_learner.*")


def main()->None:
    with open("test\predict\config.yaml", "r") as file:
        config = yaml.safe_load(file)


    model_path = config["path"]["model_path"]
    learn = load_learner(model_path)


    image_dir = config["path"]["image_dir"]
    if config["arg"]["do_recognition"]:
        recognition_character(image_dir, learn)  # 開始辨識圖片

    recognized = config["path"]["predicted_folder"]
    N_AUGMENTS_PER_IMAGE = config["N_AUGMENTS_PER_IMAGE"]
    dst_folder = config["path"]["augment_folder"]
    verified_folder = config["path"]["verified_folder"]
    if config["arg"]["do_augmentation"]:
    
        # 執行增強
        process_images(
            processed_folder=verified_folder,
            unprocessed_folder=recognized+"/has_character",
            dst_folder=dst_folder,
            N_AUGMENTS_PER_IMAGE=N_AUGMENTS_PER_IMAGE
        )
    else:
        # 若不啟動增強 → 直接將圖片移動到 verified
        unprocessed = recognized+"/has_character"
        
        for fname in os.listdir(unprocessed):
            shutil.move(os.path.join(unprocessed, fname), os.path.join(verified_folder, fname))
    
    if config["arg"]["do_training"]:
        dls = config["path"]["dataset"]
        model_training(dls, model_path)



if __name__ == "__main__":
    main()