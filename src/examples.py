"""Copy a few test examples of each error-analysis case, with their titles, for the report.

In:  <ckpt>/results/early_test_predictions.csv (src/train_fusion.py), <splits_dir>/test.csv (titles),
     <data_dir>/images/test/<class>/<image>.jpg
Out: <out_dir>/<case>/<image>.jpg and <out_dir>/examples.csv
     (case, sample_id, title, true / image / text / fusion class names, file path in out_dir)
"""
import os
import shutil
from typing import Callable

import pandas as pd

from src.data import clean_title, load_class_names

# Same cases as notebooks/analysis.ipynb (section 8.2). Folder name -> mask over the predictions.
CASES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "a_image_wrong_fusion_right": lambda p: ~p["img_ok"] & p["fus_ok"],
    "b_text_wrong_fusion_right": lambda p: ~p["txt_ok"] & p["fus_ok"],
    "c_both_wrong_fusion_right": lambda p: ~p["img_ok"] & ~p["txt_ok"] & p["fus_ok"],
}


def load_predictions(predictions_csv: str, test_csv: str) -> pd.DataFrame:
    """Predictions joined with the test split (title, class folder). Out: one row per test sample."""
    preds = pd.read_csv(predictions_csv)
    test = pd.read_csv(test_csv).rename(columns={"image": "sample_id", "label": "class_folder"})
    merged = preds.merge(test[["sample_id", "title", "class_folder"]], on="sample_id",
                         how="left", validate="one_to_one")
    if merged["class_folder"].isna().any():
        raise RuntimeError("some predicted samples are missing from test.csv")
    merged["img_ok"] = merged["image_pred"] == merged["label"]
    merged["txt_ok"] = merged["text_pred"] == merged["label"]
    merged["fus_ok"] = merged["fusion_pred"] == merged["label"]
    return merged


def export_examples(
    predictions_csv: str,
    test_csv: str,
    data_dir: str,
    out_dir: str,
    n_per_case: int,
    seed: int,
) -> pd.DataFrame:
    """Copy n_per_case random images of each case to out_dir/<case>/ and write examples.csv."""
    preds = load_predictions(predictions_csv, test_csv)
    class_names = load_class_names(data_dir)                 # index = integer label
    images_dir = os.path.join(data_dir, "images", "test")

    shutil.rmtree(out_dir, ignore_errors=True)               # re-running replaces, never accumulates
    rows = []
    for case, mask_fn in CASES.items():
        mask = mask_fn(preds)
        sample = preds[mask].sample(min(n_per_case, int(mask.sum())), random_state=seed)
        os.makedirs(os.path.join(out_dir, case), exist_ok=True)
        for _, row in sample.iterrows():
            src = os.path.join(images_dir, row["class_folder"], row["sample_id"])
            shutil.copy2(src, os.path.join(out_dir, case, row["sample_id"]))
            rows.append({
                "case": case,
                "sample_id": row["sample_id"],
                "title": clean_title(row["title"]) if isinstance(row["title"], str) else "",
                "true": class_names[row["label"]],
                "image_pred": class_names[row["image_pred"]],
                "text_pred": class_names[row["text_pred"]],
                "fusion_pred": class_names[row["fusion_pred"]],
                "file": f"{case}/{row['sample_id']}",
            })
        print(f"{case}: {int(mask.sum())} candidates, copied {len(sample)}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "examples.csv"), index=False)
    print(f"\nwrote {os.path.join(out_dir, 'examples.csv')}")
    print(table.drop(columns="file").to_string(index=False))
    return table
