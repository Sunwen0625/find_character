# Find Character

A small Python project for sorting images by whether they contain a target character, augmenting verified positive images, and retraining a FastAI classifier.

## Project layout

```text
.
├── configs/              # YAML configuration files
├── src/find_character/   # Importable Python package
├── main.py               # Backward-compatible CLI wrapper
├── train.py              # Backward-compatible training wrapper
└── pyproject.toml        # Package metadata and dependencies
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Configure

Copy or edit `configs/config.yaml` and set these important paths:

- `path.image_dir`: input images to classify.
- `path.predicted_folder`: output folder for model predictions.
- `path.verified_folder`: manually verified positive images.
- `path.augment_folder`: generated augmented positive images.
- `path.dataset_path`: folder-based FastAI dataset.
- `path.model_path`: exported FastAI model file.

The expected classifier labels are `has_character` and `no_character`.

## Run

```bash
python -m find_character --config configs/config.yaml
```

After installing the package, you can also run:

```bash
find-character --config configs/config.yaml
```

## Legacy scripts

The original top-level scripts still exist as compatibility wrappers, but new code should import from `find_character` under `src/`.
