"""
small_ann.py — Capacity-constrained ANN ablation model.

Architecture:
    Conv2d(1->16) -> ReLU -> MaxPool2x2
    Conv2d(16->32) -> ReLU -> MaxPool2x2
    Flatten
    Linear(1568->64) -> ReLU
    Linear(64->10)

Parameters: 105,866 (7.8x fewer than Full ANN at 824,458)
Energy:     computed analytically via Lemaire et al. 45nm CMOS model
            MACs x 4.6 pJ = 5138.5 nJ per inference

Design decisions:
    - Channel halving (32->16, 64->32) reduces conv capacity proportionally
    - FC halved (3136->256 becomes 1568->64) — note flattened size changes
      because conv2 now outputs 32 channels not 64: 32*7*7 = 1568
    - No Dropout — matches Full ANN inference behaviour
    - ReLU throughout — no LIF, no time dimension
    - Identical training config to Full ANN (Adam lr=1e-3, 20 epochs,
      ReduceLROnPlateau, batch=128, seeds 42/123/7)
"""

import torch
import torch.nn as nn


class SmallANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()
        self.fc1   = nn.Linear(32 * 7 * 7, 64)  # 1568 -> 64
        self.fc2   = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # [B, 16, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))  # [B, 32,  7,  7]
        x = x.view(x.size(0), -1)               # [B, 1568]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_macs():
    """
    Analytical MAC count for SmallANN.
    Conv MACs = C_out * C_in * kH * kW * H_out * W_out
    FC MACs   = in_features * out_features
    Input assumed: [1, 28, 28]
    """
    conv1_macs = 16 *  1 * 3 * 3 * 28 * 28
    conv2_macs = 32 * 16 * 3 * 3 * 14 * 14
    fc1_macs   = 1568 * 64
    fc2_macs   =   64 * 10
    return conv1_macs + conv2_macs + fc1_macs + fc2_macs


def compute_energy_nj(mac_cost_pj=4.6):
    """
    Lemaire et al. 45nm CMOS energy estimate.
    mac_cost_pj: cost per MAC in picojoules (default 4.6 pJ)
    Returns energy in nanojoules.
    """
    return compute_macs() * mac_cost_pj * 1e-3  # pJ -> nJ


if __name__ == '__main__':
    model = SmallANN()
    params = count_parameters(model)
    macs   = compute_macs()
    energy = compute_energy_nj()
    print(f'Parameters : {params:,}')
    print(f'MACs       : {macs:,}')
    print(f'Energy     : {energy:.2f} nJ')
