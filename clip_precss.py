"""Compatibility wrapper for CLIP similarity sorting."""

from find_character.clip_similarity import sort_by_clip_similarity


if __name__ == "__main__":
    sort_by_clip_similarity(
        character_img_path="test/predict/128464957_p0_master1200.jpg",
        image_folder="test/predict/input",
        output_csv="test/predict/results.csv",
        save_has_character="test/predict/result/has_character",
        save_no_character="test/predict/result/no_character",
        threshold=0.8,
    )
