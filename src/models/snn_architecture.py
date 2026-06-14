"""
snn_architecture.py
SNN model definition — identical backbone to Week 1 ANN,
ReLU replaced with snntorch Leaky LIF neurons.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNN(nn.Module):
    """
    Spiking Neural Network with Conv-LIF backbone for Fashion-MNIST.

    Architecture:
        Conv2d(1→32) → LIF → MaxPool
        Conv2d(32→64) → LIF → MaxPool
        Linear(3136→256) → LIF → Dropout
        Linear(256→10) → LIF

    Args:
        beta       : membrane decay factor for all LIF neurons (default 0.9)
        threshold  : firing threshold for all LIF neurons (default 1.0)
        slope      : fast sigmoid surrogate gradient slope (default 25)
        dropout    : dropout probability after first FC layer (default 0.3)
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: int = 25,
        dropout: float = 0.3,
    ):
        super(SNN, self).__init__()

        spike_grad = surrogate.fast_sigmoid(slope=slope)

        # ── Convolutional block 1 ──────────────────────────────────────
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.lif1  = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            init_hidden=True,
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # ── Convolutional block 2 ──────────────────────────────────────
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif2  = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            init_hidden=True,
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # ── Fully connected block ──────────────────────────────────────
        self.fc1  = nn.Linear(64 * 7 * 7, 256)   # flatten: 3136 → 256
        self.lif3 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            init_hidden=True,
        )
        self.drop = nn.Dropout(dropout)

        # ── Output layer ───────────────────────────────────────────────
        self.fc2  = nn.Linear(256, 10)
        self.lif4 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            init_hidden=True,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass over T timesteps.

        Args:
            x : spike input tensor of shape [T, B, 1, 28, 28]

        Returns:
            spk_rec : stacked output spikes  [T, B, 10]
            mem_rec : stacked output membrane [T, B, 10]
        """
        # Reset all hidden states at the start of each new sample
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()
        self.lif4.init_leaky()

        spk_rec = []
        mem_rec = []

        for t in range(x.shape[0]):
            xt = x[t]                               # [B, 1, 28, 28]

            cur1 = self.pool1(self.conv1(xt))       # [B, 32, 14, 14]
            spk1 = self.lif1(cur1)

            cur2 = self.pool2(self.conv2(spk1))     # [B, 64, 7, 7]
            spk2 = self.lif2(cur2)

            cur3 = self.fc1(spk2.flatten(1))        # [B, 256]
            spk3 = self.lif3(cur3)
            spk3 = self.drop(spk3)

            cur4 = self.fc2(spk3)                   # [B, 10]
            spk4 = self.lif4(cur4)
            mem4 = self.lif4.mem                    # membrane stored as attribute

            spk_rec.append(spk4)
            mem_rec.append(mem4)

        return torch.stack(spk_rec), torch.stack(mem_rec)

    def transfer_conv_weights(self, ann_state_dict: dict) -> list:
        """
        Copy conv1 and conv2 weights from an ANN checkpoint state dict.
        FC layers are left untouched (random init).

        Args:
            ann_state_dict : state dict from ann_baseline.pt

        Returns:
            transferred : list of key names successfully copied
        """
        snn_sd = self.state_dict()
        transferred = []

        for key in ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"]:
            if key in ann_state_dict and key in snn_sd:
                snn_sd[key] = ann_state_dict[key]
                transferred.append(key)
            else:
                print(f"  WARNING: key '{key}' not found — skipped")

        self.load_state_dict(snn_sd)
        return transferred


def build_snn(
    beta: float = 0.9,
    threshold: float = 1.0,
    slope: int = 25,
    dropout: float = 0.3,
    device: torch.device = None,
) -> SNN:
    """
    Convenience builder — instantiates SNN and moves to device.

    Args:
        beta, threshold, slope, dropout : passed to SNN.__init__
        device : torch.device; defaults to CUDA if available

    Returns:
        model : SNN instance on the requested device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SNN(beta=beta, threshold=threshold, slope=slope, dropout=dropout)
    return model.to(device)
