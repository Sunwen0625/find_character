"""Compatibility exports for image augmentation helpers."""

from find_character.augment import augment_image_and_save, get_transform, load_image, move_original_image, process_images


if __name__ == "__main__":
    process_images(
        processed_folder="test/predict/verified/has_character",
        unprocessed_folder="test/predict/temp_correct_image",
        dst_folder="test/predict/augment_positive_images/has_character",
        n_augments_per_image=2,
    )
