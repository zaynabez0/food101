"""Stratified train/val split of the UPMC Food-101 train set; the official test set is kept as is.

Reads  <data_dir>/texts/{train,test}_titles.csv   (no header: image filename, title, label)
Writes <splits_dir>/{train,val,test}.csv          (header:    image, title, label)
"""
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

COLUMNS = ["image", "title", "label"]


def load_config(path: str) -> dict:
    """Read a YAML config file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_titles(csv_path: str) -> pd.DataFrame:
    """Read a header-less UPMC titles CSV into a DataFrame with columns image, title, label."""
    return pd.read_csv(csv_path, header=None, names=COLUMNS)


def keep_existing_images(df: pd.DataFrame, images_dir: str) -> pd.DataFrame:
    """Drop rows whose image file images_dir/<label>/<image> does not exist."""
    paths = images_dir + "/" + df["label"] + "/" + df["image"]
    exists = paths.map(os.path.exists)
    print(f"{images_dir}: dropped {(~exists).sum()} rows with no image file")
    return df[exists].reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (train, val) so that every class keeps the same proportion in both."""
    train_df, val_df = train_test_split(
        df, test_size=val_fraction, stratify=df["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def summarize(name: str, df: pd.DataFrame) -> None:
    """Print size, class count, smallest/largest class and missing titles of a split."""
    per_class = df["label"].value_counts()
    print(
        f"{name:>5}: {len(df):>6} rows | {per_class.size} classes | "
        f"per class min={per_class.min()} max={per_class.max()} | "
        f"missing titles={df['title'].isna().sum()}"
    )


def make_splits(config: dict) -> None:
    """Build train/val/test CSVs from the raw UPMC files and save them to config['splits_dir']."""
    data_dir = config["data_dir"]
    train_full = load_titles(os.path.join(data_dir, "texts", "train_titles.csv"))
    test_df = load_titles(os.path.join(data_dir, "texts", "test_titles.csv"))

    train_full = keep_existing_images(train_full, os.path.join(data_dir, "images", "train"))
    test_df = keep_existing_images(test_df, os.path.join(data_dir, "images", "test"))

    train_df, val_df = stratified_split(train_full, config["val_fraction"], config["seed"])

    os.makedirs(config["splits_dir"], exist_ok=True)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.to_csv(os.path.join(config["splits_dir"], f"{name}.csv"), index=False)
        summarize(name, df)
    print(f"Saved to {config['splits_dir']}")
