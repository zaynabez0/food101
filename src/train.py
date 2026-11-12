"""Generic train / eval loop shared by every model (image, text, fusion).

- Mixed precision (AMP, float16 + GradScaler) when running on a GPU.
- Per-epoch train/val loss + accuracy written to a JSON file (config included).
- best.pt = weights with the highest val accuracy; last.pt = full state, to resume training.
Only the validation split drives decisions here (best checkpoint, LR schedule, early stopping);
the test set is never used in this file.

Expected (flat) config keys: epochs, amp, seed, batch_size, optimizer (sgd + momentum | adam), lr,
scheduler, patience, and lr_schedule or factor, plus the data.yaml keys;
optional: aux_loss_weight, early_stopping, log_every.
"""
import json
import os
import random
import tempfile
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

Batch = list[torch.Tensor]  # (image, input_ids, attention_mask, label), as returned by FoodDataset


# ---------------------------------------------------------------- setup helpers
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + GPU) so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- one batch
def forward_batch(
    model: nn.Module, batch: Batch, mode: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Move a batch to device and run the model on the inputs its mode needs.

    mode "fusion": batch = (*model_inputs, labels) from a TensorDataset of cached features,
    e.g. (image_probs (B, 101), text_probs (B, 101), labels (B,)).
    Out: logits (B, C), aux_logits (B, C) or None, labels (B,).
    """
    if mode == "fusion":
        *inputs, labels = (t.to(device, non_blocking=True) for t in batch)
        return model(*inputs), None, labels

    image, input_ids, attention_mask, labels = (t.to(device, non_blocking=True) for t in batch)
    if mode == "image":
        inputs = (image,)                                  # (B, 3, 299, 299)
    elif mode == "text":
        inputs = (input_ids, attention_mask)               # (B, L), (B, L)
    else:
        inputs = (image, input_ids, attention_mask)

    if hasattr(model, "forward_with_aux"):                 # InceptionV3: auxiliary classifier
        logits, aux_logits = model.forward_with_aux(*inputs)
    else:
        logits, aux_logits = model(*inputs), None
    return logits, aux_logits, labels


def compute_loss(
    logits: torch.Tensor, aux_logits: torch.Tensor | None, labels: torch.Tensor, aux_weight: float
) -> torch.Tensor:
    """Cross-entropy on the main logits, plus aux_weight * cross-entropy on the aux logits if any."""
    loss = nn.functional.cross_entropy(logits, labels)
    if aux_logits is not None and aux_weight > 0:
        loss = loss + aux_weight * nn.functional.cross_entropy(aux_logits, labels)
    return loss


class EpochMeter:
    """Accumulates loss and accuracy on the device; syncs with the CPU only once, at the end."""

    def __init__(self, device: torch.device) -> None:
        self.loss_sum = torch.zeros((), device=device)
        self.correct = torch.zeros((), device=device, dtype=torch.long)
        self.count = 0

    def update(self, loss: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> None:
        batch_size = labels.size(0)
        self.loss_sum += loss.detach() * batch_size         # loss is a batch mean -> sum
        self.correct += (logits.argmax(dim=1) == labels).sum()
        self.count += batch_size

    def result(self) -> dict[str, float]:
        return {"loss": self.loss_sum.item() / self.count, "acc": self.correct.item() / self.count}


# ---------------------------------------------------------------- one epoch
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mode: str,
    aux_weight: float = 0.0,
    log_every: int = 0,
) -> dict[str, float]:
    """One pass over loader with gradient updates. Out: {"loss", "acc"} averaged over the epoch."""
    model.train()                                           # dropout on, aux head on
    meter = EpochMeter(device)
    for step, batch in enumerate(loader, start=1):
        # Forward in float16 where safe (matmuls, convs); autocast keeps softmax/CE in float32.
        with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            logits, aux_logits, labels = forward_batch(model, batch, mode, device)
            loss = compute_loss(logits, aux_logits, labels, aux_weight)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()   # scale loss up so small float16 gradients don't round to 0
        scaler.step(optimizer)          # unscale grads; skip the step if any are inf/NaN
        scaler.update()                 # adapt the scale factor for the next step
        meter.update(loss, logits, labels)

        if log_every and step % log_every == 0:
            print(f"  step {step}/{len(loader)}  loss {loss.item():.4f}")
    return meter.result()


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, mode: str, use_amp: bool
) -> dict[str, float]:
    """One pass over loader without gradients. Out: {"loss", "acc"} (main logits only)."""
    model.eval()                                            # dropout off, aux head off
    meter = EpochMeter(device)
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits, _, labels = forward_batch(model, batch, mode, device)
            loss = nn.functional.cross_entropy(logits, labels)
        meter.update(loss, logits, labels)
    return meter.result()


# ---------------------------------------------------------------- checkpoints + log
def atomic_save(obj: object, path: str) -> None:
    """torch.save to a temp file, then rename: a crash mid-save never leaves a broken checkpoint."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def write_results(path: str, config: dict, history: list[dict], best_val_acc: float) -> None:
    """Write config + per-epoch metrics to a JSON file (rewritten every epoch)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"config": config, "best_val_acc": best_val_acc, "history": history}, f, indent=2)
    os.replace(tmp_path, path)


class StepListOnPlateau:
    """Move to the next LR of a fixed list when val accuracy stalls for `patience` epochs.

    Paper: the LR is lowered as epochs increase (0.01 -> 0.001 -> 0.0005). Unlike
    ReduceLROnPlateau (one constant factor), the drops here can differ (x0.1 then x0.5).
    """

    def __init__(self, optimizer: torch.optim.Optimizer, lrs: list[float], patience: int) -> None:
        self.optimizer, self.lrs, self.patience = optimizer, lrs, patience
        self.index, self.best, self.bad_epochs = 0, float("-inf"), 0
        self._set_lr(lrs[0])

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self, val_acc: float) -> None:
        if val_acc > self.best:
            self.best, self.bad_epochs = val_acc, 0
            return
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience and self.index < len(self.lrs) - 1:
            self.index, self.bad_epochs = self.index + 1, 0
            self._set_lr(self.lrs[self.index])
            print(f"  val acc stalled for {self.patience} epochs -> lr = {self.lrs[self.index]}")

    def state_dict(self) -> dict:
        return {"index": self.index, "best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict) -> None:
        self.index, self.best, self.bad_epochs = state["index"], state["best"], state["bad_epochs"]
        self._set_lr(self.lrs[self.index])


def step_scheduler(scheduler: object | None, val_acc: float) -> None:
    """Advance the LR scheduler once per epoch; plateau schedulers watch val accuracy."""
    if scheduler is None:
        return
    if isinstance(scheduler, (StepListOnPlateau, torch.optim.lr_scheduler.ReduceLROnPlateau)):
        scheduler.step(val_acc)       # ReduceLROnPlateau must be created with mode="max"
    else:
        scheduler.step()


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Optimizer over the trainable parameters only (a frozen BERT is left out), from config['optimizer']."""
    params = [p for p in model.parameters() if p.requires_grad]
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=config["lr"], momentum=config["momentum"])
    if config["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=config["lr"])
    raise ValueError(f"unsupported optimizer {config['optimizer']!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> object | None:
    """LR scheduler from config['scheduler'] ('plateau' or 'none'); both plateau kinds watch val acc.

    plateau + lr_schedule -> StepListOnPlateau (image model: explicit list of LRs)
    plateau + factor      -> ReduceLROnPlateau (text model: multiply LR by factor)
    """
    if config["scheduler"] == "none":
        return None
    if config["scheduler"] != "plateau":
        raise ValueError(f"unsupported scheduler {config['scheduler']!r}")
    if "lr_schedule" in config:
        if config["lr"] != config["lr_schedule"][0]:
            raise ValueError("config 'lr' must equal the first value of 'lr_schedule'")
        return StepListOnPlateau(optimizer, config["lr_schedule"], config["patience"])
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config["factor"], patience=config["patience"]
    )


# ---------------------------------------------------------------- full training run
def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    config: dict,
    mode: str,
    ckpt_dir: str,
    results_path: str,
    on_epoch_end: Callable[[], None] | None = None,
) -> list[dict]:
    """Train for config["epochs"] epochs, resuming from ckpt_dir/last.pt if present.

    on_epoch_end: called after each epoch's files are written (Modal uses it to commit the volume).
    Out: the per-epoch history (also saved to results_path).
    """
    device = next(model.parameters()).device
    use_amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    patience = config.get("early_stopping")                 # epochs without val improvement; None = off

    os.makedirs(ckpt_dir, exist_ok=True)
    last_path, best_path = os.path.join(ckpt_dir, "last.pt"), os.path.join(ckpt_dir, "best.pt")

    start_epoch, best_val_acc, epochs_no_improve, history = 1, 0.0, 0, []
    if os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_val_acc, epochs_no_improve = state["best_val_acc"], state["epochs_no_improve"]
        history = state["history"]
        print(f"Resumed from {last_path}: next epoch {start_epoch}, best val acc {best_val_acc:.4f}")

    for epoch in range(start_epoch, config["epochs"] + 1):
        if patience and epochs_no_improve >= patience:
            print(f"Early stopping: no val improvement for {patience} epochs.")
            break

        start_time = time.time()
        lr = optimizer.param_groups[0]["lr"]                # LR used during this epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, mode,
            config.get("aux_loss_weight", 0.0), config.get("log_every", 0),
        )
        val_metrics = evaluate(model, val_loader, device, mode, use_amp)
        step_scheduler(scheduler, val_metrics["acc"])

        if val_metrics["acc"] > best_val_acc:
            best_val_acc, epochs_no_improve = val_metrics["acc"], 0
            atomic_save({"model": model.state_dict(), "epoch": epoch, "val_acc": best_val_acc}, best_path)
        else:
            epochs_no_improve += 1

        history.append({
            "epoch": epoch, "lr": lr, "time_s": round(time.time() - start_time, 1),
            "train_loss": train_metrics["loss"], "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"],
        })
        atomic_save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "epochs_no_improve": epochs_no_improve,
            "history": history,
        }, last_path)
        write_results(results_path, config, history, best_val_acc)

        print(
            f"epoch {epoch:>3} | lr {lr:.2e} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} | "
            f"{history[-1]['time_s']}s" + ("  *best*" if epochs_no_improve == 0 else "")
        )
        if on_epoch_end is not None:
            on_epoch_end()
    return history


def train_model(
    model: nn.Module,
    config: dict,
    mode: str,
    ckpt_dir: str,
    results_path: str,
    max_items: int | None = None,
    on_epoch_end: Callable[[], None] | None = None,
) -> list[dict]:
    """Seed, build loaders/optimizer/scheduler from config, then fit. max_items: subset per split."""
    from src.data import load_tokenizer, make_dataloader  # local import: keeps the smoke test light

    set_seed(config["seed"])
    device = get_device()
    model = model.to(device)
    tokenizer = load_tokenizer(config) if mode in ("text", "both") else None
    train_loader = make_dataloader("train", mode, config, config["batch_size"], tokenizer, max_items)
    val_loader = make_dataloader("val", mode, config, config["batch_size"], tokenizer, max_items)
    print(f"device {device} | train {len(train_loader.dataset)} | val {len(val_loader.dataset)} items")

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    return fit(model, train_loader, val_loader, optimizer, scheduler, config, mode,
               ckpt_dir, results_path, on_epoch_end)


# ---------------------------------------------------------------- smoke test
def smoke_test() -> None:
    """Train a toy model on random data for 2 epochs, then 'restart' and check it resumes at epoch 3."""
    set_seed(0)
    device = get_device()
    n, n_classes = 64, 5
    images = torch.randn(n, 3, 8, 8)
    empty = torch.empty(n, 0)
    labels = torch.randint(0, n_classes, (n,))
    loader = DataLoader(TensorDataset(images, empty, empty, labels), batch_size=16)

    def make_run() -> tuple[nn.Module, torch.optim.Optimizer, object]:
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, n_classes)).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=0)
        return model, optimizer, scheduler

    with tempfile.TemporaryDirectory() as tmp:
        results_path = os.path.join(tmp, "results", "toy.json")
        for epochs in (2, 3):                               # 2nd call = restart after a "crash"
            config = {"epochs": epochs, "amp": True}
            print(f"\n--- fit(epochs={epochs}) on {device} ---")
            model, optimizer, scheduler = make_run()        # fresh objects, like a new process
            fit(model, loader, loader, optimizer, scheduler, config, "image", tmp, results_path)
        print("\nfiles:", sorted(os.listdir(tmp)))
        with open(results_path) as f:
            print("results JSON epochs:", [h["epoch"] for h in json.load(f)["history"]])
