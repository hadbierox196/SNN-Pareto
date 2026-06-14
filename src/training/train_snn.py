"""
train_snn.py
Full training, validation, and test pipeline for the SNN.
Mirrors the Week 2 Colab notebook logic as importable functions.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import spikegen

from src.models.snn_architecture import build_snn, SNN


# ── Constants ──────────────────────────────────────────────────────────────────

MEAN = 0.2860
STD  = 0.3530


# ── Data ───────────────────────────────────────────────────────────────────────

def get_dataloaders(
    batch_size: int = 128,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 2,
):
    """
    Returns train, val, and test DataLoaders for Fashion-MNIST.
    Uses the same normalization as Week 1 ANN baseline.

    Args:
        batch_size    : training batch size (eval uses 256)
        val_fraction  : fraction of train set held out for validation
        seed          : random seed for the train/val split
        num_workers   : DataLoader worker count

    Returns:
        train_loader, val_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])

    train_dataset = datasets.FashionMNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        "./data", train=False, download=True, transform=transform
    )

    val_size   = int(val_fraction * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_set, val_set = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=256,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ── Spike encoding ─────────────────────────────────────────────────────────────

def poisson_encode(imgs: torch.Tensor, T: int) -> torch.Tensor:
    """
    Convert a batch of images to Poisson spike trains.

    Args:
        imgs : image tensor [B, 1, 28, 28] (normalized, may be negative)
        T    : number of timesteps

    Returns:
        spike_input : [T, B, 1, 28, 28] binary spike tensor
    """
    # spikegen.rate expects values in [0, 1]
    imgs_norm = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
    return spikegen.rate(imgs_norm, num_steps=T)


# ── Spike rate hooks ───────────────────────────────────────────────────────────

class SpikeRateTracker:
    """
    Attaches forward hooks to all LIF layers in an SNN and
    records mean spike rate per batch. Call .epoch_mean() to
    get the average over the epoch and .reset() before each epoch.
    """

    def __init__(self, model: SNN):
        self.rates   = {name: [] for name in ["lif1", "lif2", "lif3", "lif4"]}
        self._hooks  = []

        for name in self.rates:
            layer  = getattr(model, name)
            handle = layer.register_forward_hook(self._make_hook(name))
            self._hooks.append(handle)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            # init_hidden=True → output is spike tensor directly
            spk = output
            self.rates[name].append(spk.detach().mean().item())
        return hook

    def reset(self):
        for k in self.rates:
            self.rates[k].clear()

    def epoch_mean(self) -> dict:
        return {
            k: float(np.mean(v)) if v else 0.0
            for k, v in self.rates.items()
        }

    def remove(self):
        for h in self._hooks:
            h.remove()


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: SNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    T: int,
    device: torch.device,
    max_norm: float = 1.0,
) -> float:
    """
    Run one training epoch.

    Returns:
        avg_loss : mean cross-entropy loss over all batches
    """
    model.train()
    running_loss = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        spike_input = poisson_encode(imgs, T).to(device)  # [T, B, 1, 28, 28]

        optimizer.zero_grad()

        spk_out, _ = model(spike_input)          # [T, B, 10]
        spk_sum    = spk_out.sum(dim=0)          # [B, 10] rate readout

        loss = loss_fn(spk_sum, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: SNN,
    loader: DataLoader,
    T: int,
    device: torch.device,
) -> float:
    """
    Evaluate accuracy on a DataLoader (val or test).

    Returns:
        accuracy : percentage correct (0–100)
    """
    model.eval()
    correct = 0
    total   = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        spike_input  = poisson_encode(imgs, T).to(device)

        spk_out, _ = model(spike_input)
        spk_sum    = spk_out.sum(dim=0)
        preds      = spk_sum.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return 100.0 * correct / total


# ── Main training loop ─────────────────────────────────────────────────────────

def train_snn(
    T: int = 32,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 128,
    beta: float = 0.9,
    slope: int = 25,
    dropout: float = 0.3,
    seed: int = 42,
    ann_checkpoint: str = None,
    save_path: str = "checkpoints/snn_T32_init.pt",
    device: torch.device = None,
) -> dict:
    """
    Full SNN training pipeline.

    Args:
        T              : number of timesteps for Poisson encoding
        epochs         : number of training epochs
        lr             : Adam learning rate
        batch_size     : training batch size
        beta           : LIF membrane decay
        slope          : surrogate gradient slope
        dropout        : dropout after first FC layer
        seed           : random seed
        ann_checkpoint : path to ann_baseline.pt for conv weight transfer
                         (None = train from scratch)
        save_path      : where to save the best checkpoint
        device         : torch.device (auto-detected if None)

    Returns:
        history : dict with keys
                  train_loss, val_acc, spike_rates, best_val_acc,
                  best_epoch, test_acc
    """
    # ── Setup ──────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    print(f"T      : {T}  |  epochs: {epochs}  |  lr: {lr}  |  beta: {beta}")

    # ── Data ───────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, seed=seed
    )

    # ── Model ──────────────────────────────────────────────────────────
    model = build_snn(beta=beta, slope=slope, dropout=dropout, device=device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optional: transfer conv weights from ANN baseline
    if ann_checkpoint is not None and os.path.exists(ann_checkpoint):
        ckpt   = torch.load(ann_checkpoint, map_location=device)
        ann_sd = ckpt.get("model_state_dict", ckpt)
        keys   = model.transfer_conv_weights(ann_sd)
        print(f"Transferred from ANN: {keys}")
    else:
        print("Training SNN from scratch (no ANN checkpoint provided).")

    # ── Optimizer + loss ───────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    # ── Spike rate tracker ─────────────────────────────────────────────
    tracker = SpikeRateTracker(model)

    # ── History ────────────────────────────────────────────────────────
    history = {
        "train_loss"  : [],
        "val_acc"     : [],
        "spike_rates" : {k: [] for k in ["lif1", "lif2", "lif3", "lif4"]},
        "best_val_acc": 0.0,
        "best_epoch"  : 0,
        "test_acc"    : None,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ── Header ─────────────────────────────────────────────────────────
    print(f"\n{'Epoch':>5} {'Train Loss':>12} {'Val Acc (%)':>12} "
          f"{'spk_lif1':>10} {'spk_lif2':>10} {'spk_lif3':>10} {'spk_lif4':>10}")
    print("-" * 85)

    # ── Epoch loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        tracker.reset()

        avg_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, T, device
        )
        val_acc = evaluate(model, val_loader, T, device)

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        rates = tracker.epoch_mean()
        for k in rates:
            history["spike_rates"][k].append(rates[k])

        # Save best checkpoint
        if val_acc > history["best_val_acc"]:
            history["best_val_acc"] = val_acc
            history["best_epoch"]   = epoch
            torch.save({
                "epoch"               : epoch,
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc"             : val_acc,
                "T"                   : T,
                "beta"                : beta,
            }, save_path)

        best_marker = " ← best" if epoch == history["best_epoch"] else ""
        print(
            f"{epoch:>5} {avg_loss:>12.4f} {val_acc:>11.2f}%  "
            f"{rates['lif1']:>10.4f} {rates['lif2']:>10.4f} "
            f"{rates['lif3']:>10.4f} {rates['lif4']:>10.4f}"
            f"{best_marker}"
        )

        # NaN guard
        if np.isnan(avg_loss):
            print(f"\n🚨 NaN loss at epoch {epoch}. Stopping.")
            break

        # Dead neuron warning
        if epoch >= 5 and rates["lif1"] < 0.01 and rates["lif2"] < 0.01:
            print(f"\n⚠️  Dead neuron warning at epoch {epoch}. "
                  "Consider reducing beta or increasing surrogate slope.")

    tracker.remove()

    # ── Test accuracy ──────────────────────────────────────────────────
    best_ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_acc = evaluate(model, test_loader, T, device)
    history["test_acc"] = test_acc

    print(f"\nBest val acc : {history['best_val_acc']:.2f}% at epoch {history['best_epoch']}")
    print(f"Test accuracy: {test_acc:.2f}%")

    # ── Sanity gates ───────────────────────────────────────────────────
    loss_monotone = history["train_loss"][-1] < history["train_loss"][0]
    no_nan        = not any(np.isnan(l) for l in history["train_loss"])

    print("\n--- Sanity Gates ---")
    print(f"Loss decreasing epoch 1→{epochs}: {'✅' if loss_monotone else '❌'}")
    print(f"No NaN loss:                      {'✅' if no_nan else '❌'}")
    print(f"Test accuracy > 80%:              {'✅' if test_acc > 80 else '❌'}")

    return history


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    history = train_snn(
        T              = 32,
        epochs         = 20,
        lr             = 1e-3,
        batch_size     = 128,
        beta           = 0.9,
        slope          = 25,
        dropout        = 0.3,
        seed           = 42,
        ann_checkpoint = "checkpoints/ann_baseline.pt",
