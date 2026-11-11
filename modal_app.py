"""Modal plumbing: GPU image, volumes for data/checkpoints, Kaggle secret, entrypoints.

Only infrastructure lives here; it calls into src/train.py.
"""
import modal

# --- App: groups all functions of this project under one name in Modal ---
app = modal.App("food101-fusion")

# --- Image: the software environment of the cloud container ---
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "pandas",
        "scikit-learn",
        "pyyaml",
        "kaggle>=1.8",
    )
    .add_local_python_source("src")  # copies your local src/ folder into the container
)

# --- Volumes: persistent storage ---
data_volume = modal.Volume.from_name("food101-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("food101-ckpt", create_if_missing=True)


# --- Test function: checks that the image works on a GPU ---
@app.function(image=image, gpu="T4", volumes={"/data": data_volume, "/ckpt": ckpt_volume})
def check_env():
    import torch, torchvision, transformers
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("transformers:", transformers.__version__)
    print("GPU available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0))