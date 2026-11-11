"""Modal plumbing: GPU image, volumes for data/checkpoints, Kaggle secret, entrypoints.

Only infrastructure lives here; it calls into src/train.py.
"""
import modal

# --- App: groups all functions of this project under one name in Modal ---
app = modal.App("food101-fusion")

CONFIGS_DIR = "/root/configs"  # where configs/ lives inside the container

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
    .add_local_dir("configs", remote_path=CONFIGS_DIR)  # YAML configs, attached at start (no rebuild)
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


# --- Data smoke test: one batch per mode, prints shapes (CPU only) ---
@app.function(image=image, volumes={DATA_DIR: data_volume}, cpu=4)
def test_data(config_name: str = "data.yaml") -> None:
    """Run src/data.py's smoke test with configs/<config_name>."""
    from src.data import smoke_test
    from src.split import load_config

    smoke_test(load_config(f"{CONFIGS_DIR}/{config_name}"))
