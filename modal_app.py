"""Modal plumbing: GPU image, volumes for data/checkpoints, Kaggle secret, entrypoints.

Only infrastructure lives here; it calls into src/train.py.
"""
from pathlib import Path
from typing import Callable

import modal

# --- App: groups all functions of this project under one name in Modal ---
app = modal.App("food101-fusion")

CONFIGS_DIR = "/root/configs"  # where configs/ lives inside the container
PROJECT_DIR = Path(__file__).resolve().parent  # so the app also works when imported from notebooks/

# --- Image: the software environment of the cloud container ---
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.15.0",
        "torchvision==0.29.2",
        "transformers==5.22.2",
        "pandas",
        "scikit-learn",
        "pyyaml",
        "kaggle==2.2.5",
        "six",  # kaggle 2.2.5 imports six but does not declare it as a dependency
    )
    .add_local_python_source("src")  # copies your local src/ folder into the container
    .add_local_dir(PROJECT_DIR / "configs", remote_path=CONFIGS_DIR)  # YAML configs, attached at start (no rebuild)
)

# --- Volumes: persistent storage ---
data_volume = modal.Volume.from_name("food101-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("food101-ckpt", create_if_missing=True)


# --- Test function: checks that the image works on a GPU ---
@app.function(image=image, gpu="any", volumes={"/data": data_volume, "/ckpt": ckpt_volume})
def check_env():
    import torch, torchvision, transformers
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("transformers:", transformers.__version__)
    print("GPU available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0))


# --- Data download: Kaggle -> food101-data volume (CPU only, run once) ---
KAGGLE_DATASET = "gianmarco96/upmcfood101"  # UPMC Food-101, uploaded by the paper's first author
DATA_DIR = "/data"                          # where data_volume is mounted
DOWNLOAD_TIMEOUT_S = 60 * 60                # Modal's default (300 s) is too short for a multi-GB zip

kaggle_secret = modal.Secret.from_name("kaggle-secret")
# Sets HF_TOKEN, which transformers reads by itself when downloading bert-base-uncased
# (authenticated downloads: higher rate limits, no "unauthenticated requests" warning).
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=image,
    secrets=[kaggle_secret],
    volumes={DATA_DIR: data_volume},
    timeout=DOWNLOAD_TIMEOUT_S,
)
def download_data(force: bool = False) -> None:
    """Download UPMC Food-101 from Kaggle, unzip into /data and commit the volume.

    In:  force -- re-download even if /data already contains files.
    Out: nothing; files persist in the food101-data volume, folder tree is printed.
    """
    import os
    import subprocess
    import zipfile

    if os.listdir(DATA_DIR) and not force:
        print(f"{DATA_DIR} is not empty, skipping download (use --force to redo).")
        print_tree(DATA_DIR)
        return

    # Download the zip to the container's local disk, not the volume,
    # so the archive itself is not stored permanently.
    tmp_dir = "/tmp/kaggle"
    subprocess.run(
        ["kaggle", "datasets", "download", KAGGLE_DATASET, "-p", tmp_dir],
        check=True,  # raise if kaggle fails (bad credentials, wrong slug...)
    )

    zip_path = os.path.join(tmp_dir, os.listdir(tmp_dir)[0])
    print(f"Unzipping {zip_path} into {DATA_DIR} ...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(DATA_DIR)

    data_volume.commit()  # make the new files visible to other containers / future runs
    print("Volume committed.")
    print_tree(DATA_DIR)


def print_tree(root: str, max_depth: int = 2) -> None:
    """Print root and the folders up to max_depth levels below it, with their file counts."""
    import os

    root_depth = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip("/").count("/") - root_depth
        if depth > max_depth:
            dirnames.clear()  # stop os.walk from going deeper
            continue
        dirnames.sort()
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath) or dirpath}/  ({len(filenames)} files)")

# --- Data inspection: what is actually inside the volume? (CPU only) ---
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
SPLIT_NAMES = ("train", "val", "test")
CSV_PREVIEW_ROWS = 5


@app.function(image=image, volumes={DATA_DIR: data_volume})
def inspect_data() -> None:
    """Print the /data tree (2 levels), a preview of every CSV, images per split and class count.

    In:  nothing (reads the food101-data volume).
    Out: nothing; everything is printed.
    """
    print("=== Folder tree (2 levels) ===")
    print_tree(DATA_DIR, max_depth=2)

    print("\n=== CSV files ===")
    for csv_path in find_files(DATA_DIR, (".csv",)):
        preview_csv(csv_path)

    print("\n=== Images per split ===")
    for split_dir in find_split_dirs(DATA_DIR):
        count_split(split_dir)


def find_files(root: str, extensions: tuple[str, ...]) -> list[str]:
    """Return the sorted paths of all files under root whose name ends with one of extensions."""
    import os

    matches = []
    for dirpath, _, filenames in os.walk(root):
        matches += [os.path.join(dirpath, f) for f in filenames if f.lower().endswith(extensions)]
    return sorted(matches)


def find_split_dirs(root: str) -> list[str]:
    """Return the sorted paths of all folders under root named train / val / test."""
    import os

    return sorted(
        dirpath for dirpath, _, _ in os.walk(root) if os.path.basename(dirpath) in SPLIT_NAMES
    )


def preview_csv(csv_path: str) -> None:
    """Print a CSV's shape, column names and first rows."""
    import pandas as pd

    df = pd.read_csv(csv_path, header=None)  # UPMC CSVs have no header row
    print(f"\n--- {csv_path}  shape={df.shape} ---")
    print("columns:", list(df.columns))
    print(df.head(CSV_PREVIEW_ROWS).to_string())


def count_split(split_dir: str) -> None:
    """Print how many images a split folder holds and how many class subfolders it has."""
    import os

    classes = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))
    n_images = len(find_files(split_dir, IMAGE_EXTENSIONS))
    print(f"{split_dir}: {n_images} images, {len(classes)} classes")
    print(f"  first classes: {classes[:5]}")


# --- Train/val/test split: writes /data/splits/*.csv (CPU only, run once) ---
@app.function(image=image, volumes={DATA_DIR: data_volume})
def split_data(config_name: str = "split.yaml") -> None:
    """Run src/split.py on the volume with configs/<config_name>, then commit the volume."""
    from src.split import load_config, make_splits

    make_splits(load_config(f"{CONFIGS_DIR}/{config_name}"))
    data_volume.commit()
    print("Volume committed.")


@app.function(image=image, volumes={DATA_DIR: data_volume})
def summarize_splits(config_name: str = "split.yaml") -> None:
    """Print src/split.py's summary of the existing /data/splits/*.csv (read-only, no re-split)."""
    import os

    import pandas as pd

    from src.split import load_config, summarize

    splits_dir = load_config(f"{CONFIGS_DIR}/{config_name}")["splits_dir"]
    for name in ("train", "val", "test"):
        summarize(name, pd.read_csv(os.path.join(splits_dir, f"{name}.csv")))


# --- Data smoke test: one batch per mode, prints shapes (CPU only) ---
@app.function(image=image, volumes={DATA_DIR: data_volume}, cpu=4, secrets=[hf_secret])
def test_data(config_name: str = "data.yaml") -> None:
    """Run src/data.py's smoke test with configs/<config_name>."""
    from src.data import smoke_test
    from src.split import load_config

    smoke_test(load_config(f"{CONFIGS_DIR}/{config_name}"))


# --- Image model shape check: dummy batch through ImageClassifier (CPU only) ---
@app.function(image=image, cpu=4)
def test_image_model(config_name: str = "image.yaml") -> None:
    """Run src/models/image.py's shape check with configs/<config_name>."""
    from src.models.image import shape_check
    from src.split import load_config

    shape_check(load_config(f"{CONFIGS_DIR}/{config_name}"))


# --- Training loop smoke test: toy model, 2 epochs, then resume (cheap GPU to exercise AMP) ---
@app.function(image=image, gpu="any")
def test_train() -> None:
    """Run src/train.py's smoke test."""
    from src.train import smoke_test

    smoke_test()


# --- Image model training: InceptionV3 on UPMC Food-101 images ---
CKPT_DIR = "/ckpt"                       # where ckpt_volume is mounted
TRAIN_TIMEOUT_S = 8 * 60 * 60            # 8 hours (Modal's maximum is 24 h)


@app.function(
    image=image,
    gpu=["L4", "A10G","T4"],  # Modal will pick any available GPU from this list
    cpu=8,                               # matches num_workers: 8 in image.yaml
    timeout=TRAIN_TIMEOUT_S,
    volumes={DATA_DIR: data_volume, CKPT_DIR: ckpt_volume},
)
def train_image(smoke: bool = False, config_name: str = "image.yaml") -> None:
    """Train the image model with configs/data.yaml + configs/<config_name>.

    smoke: 2 epochs on 500 images per split, in a separate run that always starts fresh.
    Re-running without --smoke resumes from /ckpt/<run_name>/last.pt if it exists.
    """
    from src.models.image import build_image_model

    run_training(config_name, "image", build_image_model, smoke)


# --- Text model training: BERT + LSTM on UPMC Food-101 titles ---
@app.function(
    image=image,
    gpu=["L4", "A10G", "T4"],  # Modal will pick any available GPU from this list
    cpu=4,                     # text batches are pre-tokenized tensors: little CPU work
    timeout=TRAIN_TIMEOUT_S,
    volumes={DATA_DIR: data_volume, CKPT_DIR: ckpt_volume},
    secrets=[hf_secret],
)
def train_text(smoke: bool = False, config_name: str = "text.yaml") -> None:
    """Train the text model with configs/data.yaml + configs/<config_name> (same options as train_image)."""
    from src.models.text import build_text_model

    run_training(config_name, "text", build_text_model, smoke)


def run_training(config_name: str, mode: str, build_model: Callable[[dict], object], smoke: bool) -> None:
    """Shared body of train_image / train_text; runs inside the Modal container.

    Merges data.yaml + configs/<config_name>, names the run after the config file
    ("image", "text", or "<name>_smoke"), builds the model and calls src.train.train_model.
    """
    import os
    import shutil

    from src.split import load_config
    from src.train import train_model

    # the model config overrides data.yaml where they share a key (e.g. num_workers, max_length)
    config = {**load_config(f"{CONFIGS_DIR}/data.yaml"), **load_config(f"{CONFIGS_DIR}/{config_name}")}
    run_name = os.path.splitext(config_name)[0]
    max_items = None
    if smoke:
        run_name += "_smoke"
        config["epochs"], max_items = config["smoke_epochs"], config["smoke_max_items"]
        shutil.rmtree(f"{CKPT_DIR}/{run_name}", ignore_errors=True)  # smoke never resumes

    print(f"run '{run_name}' (mode={mode}): {config['epochs']} epochs, max_items={max_items}")
    train_model(
        build_model(config),
        config,
        mode=mode,
        ckpt_dir=f"{CKPT_DIR}/{run_name}",
        results_path=f"{CKPT_DIR}/results/{run_name}.json",
        max_items=max_items,
        on_epoch_end=ckpt_volume.commit,  # persist checkpoints + JSON after every epoch
    )


# --- Text model shape check: dummy batch through TextClassifier (CPU only) ---
@app.function(image=image, cpu=4, secrets=[hf_secret])
def test_text_model(config_name: str = "text.yaml") -> None:
    """Run src/models/text.py's shape check with configs/data.yaml + configs/<config_name>."""
    from src.models.text import shape_check
    from src.split import load_config

    shape_check({**load_config(f"{CONFIGS_DIR}/data.yaml"), **load_config(f"{CONFIGS_DIR}/{config_name}")})


# --- Phase 4 feature caching: best (or given) checkpoints -> /ckpt/features/{split}.pt ---
@app.function(
    image=image,
    gpu=["L4", "A10G", "T4"],
    cpu=8,                     # matches num_workers: 8 in extract.yaml (JPEG decoding)
    timeout=TRAIN_TIMEOUT_S,
    volumes={DATA_DIR: data_volume, CKPT_DIR: ckpt_volume},
    secrets=[hf_secret],
)
def extract_features(image_ckpt: str = "", text_ckpt: str = "", splits: str = "", smoke: bool = False) -> None:
    """Run src/extract.py. image_ckpt / text_ckpt override the best.pt paths of configs/extract.yaml;
    splits (e.g. "test" or "val,test") overrides its list of splits.
    smoke: only the first smoke_max_items rows per split, saved to smoke_features_dir."""
    from src.extract import run_extraction
    from src.models.image import build_image_model
    from src.models.text import build_text_model
    from src.split import load_config

    data = load_config(f"{CONFIGS_DIR}/data.yaml")
    image_cfg = {**data, **load_config(f"{CONFIGS_DIR}/image.yaml")}
    text_cfg = {**data, **load_config(f"{CONFIGS_DIR}/text.yaml")}
    # dataset/tokenizer settings come from text.yaml (same max_length as in training), loop from extract.yaml
    config = {**text_cfg, **load_config(f"{CONFIGS_DIR}/extract.yaml")}
    if image_ckpt:
        config["image_checkpoint"] = image_ckpt
    if text_ckpt:
        config["text_checkpoint"] = text_ckpt
    if splits:
        config["splits"] = splits.split(",")
    if smoke:
        config["max_items"] = config["smoke_max_items"]
        config["features_dir"] = config["smoke_features_dir"]

    run_extraction(config, build_image_model(image_cfg), build_text_model(text_cfg),
                   on_split_done=ckpt_volume.commit)


# --- Late fusion (Table III): MLP over the cached softmax vectors, CPU only ---
FUSION_TIMEOUT_S = 60 * 60               # 1 hour: ~61k rows of 2 x 101 floats, a few seconds per epoch


@app.function(
    image=image,
    cpu=4,                               # no GPU: ~98k parameters, inputs already in memory
    timeout=FUSION_TIMEOUT_S,
    volumes={CKPT_DIR: ckpt_volume},     # features, checkpoints and results all live in /ckpt
)
def train_late(merge: str = "concat", config_name: str = "late_fusion.yaml") -> None:
    """Train LateFusion with merge = concat | sum | max | weighted on /ckpt/features/{split}.pt.

    Writes /ckpt/late_{merge}/{best,last}.pt and /ckpt/results/late_{merge}.json (with test metrics).
    """
    from src.split import load_config
    from src.train_fusion import train_late_fusion

    config = {**load_config(f"{CONFIGS_DIR}/{config_name}"), "merge": merge}
    train_late_fusion(config, CKPT_DIR, on_epoch_end=ckpt_volume.commit)
    ckpt_volume.commit()                 # final JSON with the test metrics


# --- No-training baselines on the cached softmax: image alone, text alone, (p_img + p_txt) / 2 ---
@app.function(image=image, cpu=2, volumes={CKPT_DIR: ckpt_volume})
def eval_baselines(config_name: str = "late_fusion.yaml") -> None:
    """Accuracy on val and test of image-only, text-only and averaged softmax -> /ckpt/results/baselines.json."""
    from src.split import load_config
    from src.train_fusion import evaluate_baselines

    config = load_config(f"{CONFIGS_DIR}/{config_name}")   # only features_dir is used
    evaluate_baselines(config["features_dir"], f"{CKPT_DIR}/results/baselines.json")
    ckpt_volume.commit()


# --- Early fusion (Table IV): MLP over the cached image (2048) + text (128) features, CPU only ---
@app.function(
    image=image,
    cpu=4,                               # no GPU: ~354k parameters, inputs already in memory
    timeout=FUSION_TIMEOUT_S,
    volumes={CKPT_DIR: ckpt_volume},
)
def train_early(config_name: str = "early_fusion.yaml", seed: int = -1) -> None:
    """Train EarlyFusion on /ckpt/features/{split}.pt.

    Writes /ckpt/early/{best,last}.pt, /ckpt/results/early.json and
    /ckpt/results/early_test_predictions.csv (id, label, image / text / early fusion prediction).
    seed >= 0: overrides the config seed and names the run early_seed<seed> (does not overwrite "early").
    """
    from src.split import load_config
    from src.train_fusion import train_early_fusion

    config = load_config(f"{CONFIGS_DIR}/{config_name}")
    run_name = "early"
    if seed >= 0:
        config["seed"], run_name = seed, f"early_seed{seed}"
    train_early_fusion(config, CKPT_DIR, on_epoch_end=ckpt_volume.commit, run_name=run_name)
    ckpt_volume.commit()                 # final JSON + CSV


# --- Error-analysis examples: random test images of cases (a), (b), (c), with their titles ---
@app.function(image=image, cpu=2, volumes={DATA_DIR: data_volume, CKPT_DIR: ckpt_volume})
def export_examples(n_per_case: int = 4, seed: int = 0) -> None:
    """Copy n_per_case images per case to /ckpt/examples/<case>/ + /ckpt/examples/examples.csv.

    Download with: modal volume get food101-ckpt examples results/
    """
    from src.examples import export_examples as run_export
    from src.split import load_config

    data = load_config(f"{CONFIGS_DIR}/data.yaml")
    run_export(
        predictions_csv=f"{CKPT_DIR}/results/early_test_predictions.csv",
        test_csv=f"{data['splits_dir']}/test.csv",
        data_dir=data["data_dir"],
        out_dir=f"{CKPT_DIR}/examples",
        n_per_case=n_per_case,
        seed=seed,
    )
    ckpt_volume.commit()


# --- Inference: a fusion model (default: best on val) on new image + title pairs ---
INFERENCE_KWARGS = dict(
    image=image,
    cpu=4,                               # a few samples: CPU is enough, and no GPU queue
    timeout=15 * 60,
    volumes={DATA_DIR: data_volume, CKPT_DIR: ckpt_volume},
    secrets=[hf_secret],                 # bert-base-uncased config + tokenizer
)


def build_predictor(run_name: str):
    """FusionPredictor for run_name ("" = best fusion run on validation); runs inside the container."""
    from src.data import load_class_names
    from src.predict import FusionPredictor
    from src.split import load_config

    data = load_config(f"{CONFIGS_DIR}/data.yaml")
    return FusionPredictor(
        image_cfg={**data, **load_config(f"{CONFIGS_DIR}/image.yaml")},
        text_cfg={**data, **load_config(f"{CONFIGS_DIR}/text.yaml")},
        ckpt_root=CKPT_DIR,
        class_names=load_class_names(data["data_dir"]),
        run_name=run_name or None,
    )


@app.function(**INFERENCE_KWARGS)
def predict_samples(images: list[bytes], titles: list[str], top_k: int = 3, run_name: str = "") -> dict:
    """Raw image files + titles -> {model, val_acc, test_acc, predictions: [top-k of image/text/fusion]}."""
    predictor = build_predictor(run_name)
    return {
        "model": predictor.run_name,
        "val_acc": predictor.val_acc,
        "test_acc": predictor.test_acc,
        "predictions": predictor.predict(images, titles, top_k),
    }


@app.function(**INFERENCE_KWARGS)
def predict_test_samples(
    sample_ids: list[str] | None = None, n_random: int = 4, seed: int = 0, top_k: int = 3, run_name: str = ""
) -> dict:
    """Test samples read from the volume (given ids, or n_random random ones) -> same output as
    predict_samples, plus "samples": [{sample_id, title, true, image bytes}] to display them."""
    from src.predict import load_test_samples
    from src.split import load_config

    data = load_config(f"{CONFIGS_DIR}/data.yaml")
    samples = load_test_samples(data["splits_dir"], data["data_dir"], sample_ids or [], n_random, seed)
    predictor = build_predictor(run_name)
    predictions = predictor.predict([s["image"] for s in samples], [s["title"] for s in samples], top_k)
    return {
        "model": predictor.run_name,
        "val_acc": predictor.val_acc,
        "test_acc": predictor.test_acc,
        "predictions": predictions,
        "samples": samples,
    }
