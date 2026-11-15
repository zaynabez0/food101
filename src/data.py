"""Data: FoodDataset, image transforms and BERT tokenization for UPMC Food-101.

Reads the split CSVs written by src/split.py (<splits_dir>/{train,val,test}.csv, columns
image, title, label). Each item is (image, input_ids, attention_mask, label).
"""
import html
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2
from transformers import AutoTokenizer, PreTrainedTokenizerBase

MODES = ("image", "text", "both")
# CHOICE: the unused modality is returned as an empty tensor so every mode keeps the same
# 4-tuple and PyTorch's default collate still works (it batches to shape (B, 0)).
EMPTY = torch.empty(0)


def load_class_names(data_dir: str) -> list[str]:
    """Return the 101 class names, sorted; index in this list = integer label."""
    return sorted(os.listdir(os.path.join(data_dir, "images", "train")))


def load_tokenizer(config: dict) -> PreTrainedTokenizerBase:
    """Load the BERT tokenizer named in the config (downloaded from the Hugging Face Hub)."""
    return AutoTokenizer.from_pretrained(config["tokenizer_name"])


def build_image_transform(config: dict, train: bool) -> v2.Compose:
    """PIL image -> normalized float tensor (3, 299, 299). Random crop + flip only if train."""
    size = config["image_size"]
    if train:
        # Data augmentation: random crop + horizontal flip.
        resize_ops = [
            v2.RandomResizedCrop(size, scale=tuple(config["crop_scale"]), antialias=True),
            v2.RandomHorizontalFlip(),
        ]
    else:
        resize_ops = [v2.Resize(config["eval_resize"], antialias=True), v2.CenterCrop(size)]
    return v2.Compose([
        v2.ToImage(),                           # PIL -> uint8 tensor (3, H, W)
        *resize_ops,                            # -> (3, 299, 299)
        v2.ToDtype(torch.float32, scale=True),  # [0, 255] -> [0, 1]
        v2.Normalize(config["imagenet_mean"], config["imagenet_std"]),
    ])


def clean_title(title: str) -> str:
    """Decode HTML entities ("&#8217;" -> "'") and strip the '[["...."]]' scraping wrapper."""
    # CHOICE: ~4% of titles are wrapped in [["..."]] (a scraping artifact); we remove it so
    # BERT does not see it as content. The paper does not describe any text cleaning.
    title = html.unescape(str(title))
    return title.removeprefix('[["').removesuffix('"]]').strip()


def tokenize_titles(
    titles: list[str], tokenizer: PreTrainedTokenizerBase, max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize all titles at once. Out: input_ids (N, L), attention_mask (N, L), L = max_length.""" # N number of titles
    encoded = tokenizer(
        titles,
        padding="max_length",   # every title padded to the same length L -> easy batching
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    n_full = int(encoded["attention_mask"][:, -1].sum())  # last position used -> may be truncated
    print(f"tokenized {len(titles)} titles, {n_full} reach max_length={max_length}")
    return encoded["input_ids"], encoded["attention_mask"]


class FoodDataset(Dataset):
    """UPMC Food-101 split. Item = (image (3,299,299), input_ids (L,), attention_mask (L,), label ())."""

    def __init__(
        self,
        split: str,
        mode: str,
        config: dict,
        tokenizer: PreTrainedTokenizerBase | None = None,
        augment: bool | None = None,
    ) -> None:
        """split: "train" / "val" / "test"; mode: "image" / "text" / "both".

        augment: random crop + flip on images; None = only for the train split.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.use_image = mode in ("image", "both")
        self.use_text = mode in ("text", "both")
        augment = (split == "train") if augment is None else augment

        df = pd.read_csv(os.path.join(config["splits_dir"], f"{split}.csv"))
        self.sample_ids = df["image"].tolist()  # image filename, e.g. "apple_pie_107.jpg"; unique id
        class_to_idx = {name: i for i, name in enumerate(load_class_names(config["data_dir"]))}
        self.labels = torch.tensor(df["label"].map(class_to_idx).to_numpy())  # (N,) int64

        if self.use_image:  # text mode: no image paths, no transforms, no file is ever opened
            # val rows were carved from the official train set, so their images live in images/train
            image_folder = "test" if split == "test" else "train"
            images_dir = os.path.join(config["data_dir"], "images", image_folder)
            self.image_paths = [
                os.path.join(images_dir, label, image) for label, image in zip(df["label"], df["image"])
            ]
            self.transform = build_image_transform(config, train=augment)

        if self.use_text:
            tokenizer = tokenizer or load_tokenizer(config)
            titles = [clean_title(t) for t in df["title"]]
            self.input_ids, self.attention_mask = tokenize_titles(
                titles, tokenizer, config["max_length"]
            )  # (N, L), (N, L)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = EMPTY
        if self.use_image:
            with Image.open(self.image_paths[i]) as img:
                image = self.transform(img.convert("RGB"))  # (3, 299, 299); convert: some are gray/RGBA
        input_ids, attention_mask = EMPTY, EMPTY
        if self.use_text:
            input_ids, attention_mask = self.input_ids[i], self.attention_mask[i]  # (L,), (L,)
        return image, input_ids, attention_mask, self.labels[i]


def make_dataloader(
    split: str,
    mode: str,
    config: dict,
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase | None = None,
    max_items: int | None = None,
) -> DataLoader:
    """DataLoader over one split; shuffled only for train. max_items keeps only the first N items."""
    dataset = FoodDataset(split, mode, config, tokenizer)
    if max_items is not None:
        # split CSVs are already shuffled by train_test_split, so the first N cover many classes
        dataset = Subset(dataset, range(min(max_items, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=config["num_workers"],
    )


def smoke_test(config: dict) -> None:
    """Load one train batch in each mode and print the shapes, dtypes and a decoded title."""
    tokenizer = load_tokenizer(config)
    for mode in MODES:
        loader = make_dataloader("train", mode, config, config["smoke_test_batch_size"], tokenizer)
        image, input_ids, attention_mask, label = next(iter(loader))
        print(f"\n[mode={mode}] {len(loader.dataset)} items")
        print(f"  image          {tuple(image.shape)} {image.dtype}")
        print(f"  input_ids      {tuple(input_ids.shape)} {input_ids.dtype}")
        print(f"  attention_mask {tuple(attention_mask.shape)} {attention_mask.dtype}")
        print(f"  label          {tuple(label.shape)} {label.dtype} {label.tolist()}")
        if mode == "both":
            print(f"  image mean/std {image.mean():.3f} / {image.std():.3f}")
            print(f"  title[0]       {tokenizer.decode(input_ids[0], skip_special_tokens=True)!r}")
