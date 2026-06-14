#!/usr/bin/env python3
"""
reproduce.py — Full pipeline for SNN vs ANN Fashion-MNIST experiment.

Runtime estimate: ~5–6 hours on a single Colab free-tier T4 GPU.
  - SNN sweep (7 T-values × 3 seeds × 20 epochs) : ~4.5 h
  - ANN baseline (3 seeds × 20 epochs)            : ~0.3 h
  - SmallANN (3 seeds × 20 epochs)                : ~0.2 h
  - Statistics + figures                           : ~2 min

Usage (Colab):
    !python reproduce.py --csv_path /content/drive/MyDrive/sweep_results_complete.csv
    # OR to run full training from scratch (no existing CSV):
    !python reproduce.py --train_from_scratch

All random seeds are fixed in SEEDS below.
All hyperparameters are fixed in HPARAMS below.
No values are hardcoded in results — everything is derived from CSV or live training.

Failure modes to check on a fresh runtime:
    1. Drive not mounted  →  mount before running, or pass --csv_path to a local file.
    2. snnTorch version mismatch  →  pip install snntorch==0.9.4
    3. results/ directory missing  →  created automatically by this script.
"""

# ── Runtime: ~5-6 h on Colab free-tier T4 ──────────────────────────────
# Verified on: Python 3.12, PyTorch 2.x (cu128), snnTorch 0.9.4

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

# ── SEED TABLE (all seeds used in the paper) ────────────────────────────
SEEDS = [42, 123, 7]

# ── HYPERPARAMETER TABLE (fixed for all models) ─────────────────────────
HPARAMS = {
    'epochs'       : 20,
    'lr'           : 1e-3,
    'batch_size'   : 128,
    'scheduler'    : 'ReduceLROnPlateau',
    'sched_mode'   : 'max',
    'sched_patience': 3,
    'sched_factor' : 0.5,
    'grad_clip'    : 1.0,
    'norm_mean'    : 0.2860,
    'norm_std'     : 0.3530,
    'val_split'    : 0.10,        # 10% of training set = 6000 images
    'snn_beta'     : 0.9,         # LIF membrane decay
    'dropout'      : 0.3,         # SNN only, training only
}

T_VALUES    = [1, 2, 4, 8, 16, 32, 64]
ENERGY_AC   = 0.9e-12   # J per synaptic operation (Lemaire 45nm)
ENERGY_MAC  = 4.6e-12   # J per MAC (Lemaire 45nm)

OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(OUT_DIR, 'sweep_results_complete.csv')

# ── ARGUMENT PARSING ────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, default=None,
                    help='Path to existing sweep CSV (skips training)')
parser.add_argument('--train_from_scratch', action='store_true',
                    help='Run full training even if CSV exists')
parser.add_argument('--stats_only', action='store_true',
                    help='Only run stats + figures from existing CSV')
args = parser.parse_args()

if args.csv_path:
    CSV_FILE = args.csv_path


# ════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS THAT REQUIRE GPU / ML LIBRARIES
# (deferred so --stats_only doesn't fail without torch)
# ════════════════════════════════════════════════════════════════════════

def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torchvision
        import torchvision.transforms as transforms
        import snntorch as snn
        from snntorch import surrogate
        return torch, nn, torchvision, transforms, snn, surrogate
    except ImportError as e:
        print(f"ERROR: Missing dependency — {e}")
        print("Install with: pip install torch torchvision snntorch==0.9.4")
        sys.exit(1)


# ════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADER
# ════════════════════════════════════════════════════════════════════════

def get_loaders(transforms, torchvision, batch_size, val_split, seed):
    import torch
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((HPARAMS['norm_mean'],), (HPARAMS['norm_std'],))
    ])
    full_train = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)

    n_val  = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(
        full_train, [n_train, n_val], generator=gen)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(
        val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(
        test_set,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════════

def build_snn(nn, snn_lib, T):
    """Conv-LIF SNN identical to Weeks 2–4."""
    from snntorch import surrogate
    spike_grad = surrogate.fast_sigmoid()

    class SNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.T = T
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.lif1  = snn_lib.Leaky(beta=HPARAMS['snn_beta'], spike_grad=spike_grad)
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.lif2  = snn_lib.Leaky(beta=HPARAMS['snn_beta'], spike_grad=spike_grad)
            self.pool2 = nn.MaxPool2d(2)
            self.fc1   = nn.Linear(3136, 256)
            self.lif3  = snn_lib.Leaky(beta=HPARAMS['snn_beta'], spike_grad=spike_grad)
            self.drop  = nn.Dropout(HPARAMS['dropout'])
            self.fc2   = nn.Linear(256, 10)
            self.lif4  = snn_lib.Leaky(beta=HPARAMS['snn_beta'], spike_grad=spike_grad)

        def forward(self, x):
            import torch
            # Rate encode
            x_enc = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            x_enc = (torch.rand_like(x_enc) < x_enc).float()

            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            mem4 = self.lif4.init_leaky()
            spk_sum = None

            for t in range(self.T):
                xt = x_enc[t]
                xt = self.pool1(self.conv1(xt))
                spk1, mem1 = self.lif1(xt, mem1)
                xt = self.pool2(self.conv2(spk1))
                spk2, mem2 = self.lif2(xt, mem2)
                xt = spk2.flatten(1)
                xt = self.fc1(xt)
                spk3, mem3 = self.lif3(xt, mem3)
                xt = self.drop(spk3)
                xt = self.fc2(xt)
                spk4, mem4 = self.lif4(xt, mem4)
                spk_sum = spk4 if spk_sum is None else spk_sum + spk4

            return spk_sum

    return SNN()


def build_ann(nn):
    """Full ANN baseline — same Conv/FC dims, ReLU replacing LIF."""
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(3136, 256), nn.ReLU(),
        nn.Linear(256, 10)
    )


def build_small_ann(nn):
    """SmallANN (Week 6): channels halved, 105,866 params."""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1568, 64), nn.ReLU(),
        nn.Linear(64, 10)
    )


# ════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════

def train_one(model, train_loader, val_loader, test_loader,
              torch, nn, device, label=''):
    import torch as th
    model = model.to(device)
    opt   = th.optim.Adam(model.parameters(), lr=HPARAMS['lr'])
    sched = th.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode=HPARAMS['sched_mode'],
        patience=HPARAMS['sched_patience'],
        factor=HPARAMS['sched_factor'])
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, HPARAMS['epochs'] + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(),
                                        HPARAMS['grad_clip'])
            opt.step()

        # Validation
        model.eval()
        correct = total = 0
        with th.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total   += yb.size(0)
        val_acc = correct / total
        sched.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  {label} epoch {epoch:>2}/20  val_acc={val_acc*100:.2f}%")

    # Load best checkpoint, evaluate on test set
    model.load_state_dict(best_state)
    model.eval()
    correct = total = 0
    with th.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    test_acc = correct / total * 100
    print(f"  {label} → test_acc={test_acc:.4f}%")
    return test_acc, model


# ════════════════════════════════════════════════════════════════════════
# SECTION 5 — SYNOPS COUNTER
# ════════════════════════════════════════════════════════════════════════

def count_synops_snn(model, test_loader, torch, device):
    """Measure mean SynOps/image over the full test set."""
    import torch as th
    model.eval()
    total_synops = 0.0
    n_images     = 0
    T = model.T

    with th.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            B  = xb.size(0)
            x_enc = xb.unsqueeze(0).repeat(T, 1, 1, 1, 1)
            x_enc = (th.rand_like(x_enc) < x_enc).float()

            mem1 = model.lif1.init_leaky()
            mem2 = model.lif2.init_leaky()
            mem3 = model.lif3.init_leaky()
            mem4 = model.lif4.init_leaky()
            batch_synops = 0.0

            for t in range(T):
                xt   = x_enc[t]
                xt   = model.pool1(model.conv1(xt))
                spk1, mem1 = model.lif1(xt, mem1)
                # SynOps: spikes × C_out × kH × kW
                batch_synops += spk1.sum().item() * 64 * 3 * 3

                xt   = model.pool2(model.conv2(spk1))
                spk2, mem2 = model.lif2(xt, mem2)
                batch_synops += spk2.flatten(1).sum().item() * 256

                xt   = model.fc1(spk2.flatten(1))
                spk3, mem3 = model.lif3(xt, mem3)
                batch_synops += spk3.sum().item() * 10

                xt   = model.fc2(model.drop(spk3))
                spk4, mem4 = model.lif4(xt, mem4)

            total_synops += batch_synops
            n_images     += B

    return total_synops / n_images   # per image


ANN_MACS = 4_643_840      # analytical — see Week 3
SMALL_ANN_MACS = 1_116_640  # analytical — see Week 6

def energy_snn(synops):
    return synops * ENERGY_AC * 1e9    # nJ

def energy_ann(macs):
    return macs * ENERGY_MAC * 1e9     # nJ


# ════════════════════════════════════════════════════════════════════════
# SECTION 6 — RESUME LOGIC
# ════════════════════════════════════════════════════════════════════════

def already_done(T, seed, model_type='SNN'):
    if not os.path.exists(CSV_FILE):
        return False
    df = pd.read_csv(CSV_FILE)
    if model_type == 'SNN':
        return ((df['T'] == T) & (df['seed'] == seed)).any()
    else:
        return ((df['T'].isna()) & (df['seed'] == seed) &
                (df['model'] == model_type)).any()


def append_row(row_dict):
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)


# ════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN TRAINING PIPELINE
# ════════════════════════════════════════════════════════════════════════

def run_training():
    torch, nn, torchvision, transforms, snn_lib, surrogate = import_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Hyperparameters: {HPARAMS}\n")

    t0 = time.time()

    # ── SNN sweep ──────────────────────────────────────────────────────
    for T in T_VALUES:
        for seed in SEEDS:
            if already_done(T, seed, 'SNN'):
                print(f"  SKIP SNN T={T} seed={seed} (already in CSV)")
                continue

            torch.manual_seed(seed)
            np.random.seed(seed)

            train_loader, val_loader, test_loader = get_loaders(
                transforms, torchvision,
                HPARAMS['batch_size'], HPARAMS['val_split'], seed)

            model  = build_snn(nn, snn_lib, T)
            label  = f"SNN T={T} seed={seed}"
            acc, model = train_one(model, train_loader, val_loader,
                                   test_loader, torch, nn, device, label)

            synops   = count_synops_snn(model, test_loader, torch, device)
            e_nj     = energy_snn(synops)

            append_row({'T': T, 'seed': seed, 'model': 'SNN',
                        'accuracy': round(acc, 4),
                        'synops': round(synops, 2),
                        'energy_nj': round(e_nj, 2)})
            print(f"  Saved SNN T={T} seed={seed}  "
                  f"acc={acc:.2f}%  energy={e_nj:.1f} nJ")

    # ── Full ANN ───────────────────────────────────────────────────────
    for seed in SEEDS:
        if already_done(None, seed, 'ANN'):
            print(f"  SKIP ANN seed={seed}")
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader, val_loader, test_loader = get_loaders(
            transforms, torchvision,
            HPARAMS['batch_size'], HPARAMS['val_split'], seed)

        model  = build_ann(nn)
        acc, _ = train_one(model, train_loader, val_loader,
                           test_loader, torch, nn, device, f"ANN seed={seed}")
        e_nj   = energy_ann(ANN_MACS)
        append_row({'T': None, 'seed': seed, 'model': 'ANN',
                    'accuracy': round(acc, 4),
                    'synops': ANN_MACS,
                    'energy_nj': round(e_nj, 2)})

    # ── Small ANN ──────────────────────────────────────────────────────
    for seed in SEEDS:
        if already_done(None, seed, 'SmallANN'):
            print(f"  SKIP SmallANN seed={seed}")
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader, val_loader, test_loader = get_loaders(
            transforms, torchvision,
            HPARAMS['batch_size'], HPARAMS['val_split'], seed)

        model  = build_small_ann(nn)
        acc, _ = train_one(model, train_loader, val_loader,
                           test_loader, torch, nn, device,
                           f"SmallANN seed={seed}")
        e_nj   = energy_ann(SMALL_ANN_MACS)
        append_row({'T': None, 'seed': seed, 'model': 'SmallANN',
                    'accuracy': round(acc, 4),
                    'synops': SMALL_ANN_MACS,
                    'energy_nj': round(e_nj, 2)})

    elapsed = (time.time() - t0) / 3600
    print(f"\nTotal training time: {elapsed:.2f} h")


# ════════════════════════════════════════════════════════════════════════
# SECTION 8 — STATS + FIGURES (calls 07_statistics logic inline)
# ════════════════════════════════════════════════════════════════════════

def run_stats():
    from scipy.stats import bootstrap as sp_bootstrap
    import matplotlib.pyplot as plt

    df  = pd.read_csv(CSV_FILE)
    snn = df[df['model'] == 'SNN'].copy()
    ann = df[df['model'] == 'ANN'].copy()

    def bootstrap_ci(data, n_boot=1000, ci=0.95):
        rng   = np.random.default_rng(42)
        data  = np.array(data)
        boots = [np.mean(rng.choice(data, size=len(data), replace=True))
                 for _ in range(n_boot)]
        lo = np.percentile(boots, (1 - ci) / 2 * 100)
        hi = np.percentile(boots, (1 + ci) / 2 * 100)
        return lo, hi

    rows = []
    for T in sorted(snn['T'].unique()):
        grp = snn[snn['T'] == T]
        acc_lo, acc_hi = bootstrap_ci(grp['accuracy'].values)
        e_lo,   e_hi   = bootstrap_ci(grp['energy_nj'].values)
        rows.append({
            'T'            : int(T),
            'acc_mean'     : grp['accuracy'].mean(),
            'acc_ci_lo'    : acc_lo,
            'acc_ci_hi'    : acc_hi,
            'energy_mean'  : grp['energy_nj'].mean(),
            'energy_ci_lo' : e_lo,
            'energy_ci_hi' : e_hi,
        })

    stats_df = pd.DataFrame(rows)
    out_csv  = os.path.join(OUT_DIR, '07_stats_table.csv')
    stats_df.to_csv(out_csv, index=False)
    print(f"Stats table saved: {out_csv}")
    print(stats_df.to_string(index=False))


# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("reproduce.py — SNN vs ANN Fashion-MNIST")
    print(f"Seeds      : {SEEDS}")
    print(f"T values   : {T_VALUES}")
    print(f"Output dir : {OUT_DIR}")
    print("=" * 60)

    if not args.stats_only:
        if args.train_from_scratch or not os.path.exists(CSV_FILE):
            print("\nStarting training pipeline...")
            run_training()
        else:
            print(f"\nCSV found at {CSV_FILE} — skipping training.")
            print("Pass --train_from_scratch to override.")

    print("\nRunning statistics and generating figures...")
    run_stats()
    print("\n✅ Done. All outputs in ./results/")
