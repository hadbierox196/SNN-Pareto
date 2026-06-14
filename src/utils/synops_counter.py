"""
src/utils/synops_counter.py

SynOps counting utilities for spiking neural networks.
Implements the formula:
    SynOps = Σ_t Σ_b Σ_i  s_i(t,b) × fan_out_i

Reference:
    Lemaire et al. (2022) "An Analytical Estimation of Spiking Neural
    Networks Energy Efficiency." https://arxiv.org/abs/2209.10074
"""

import torch
import torch.nn as nn


def count_synops(spk_rec: torch.Tensor, weight: torch.Tensor) -> float:
    """
    Count synaptic operations for a fully-connected layer.

    Args:
        spk_rec : Tensor [T, B, N_pre]   spike trains from pre-synaptic layer
        weight  : Tensor [N_post, N_pre]  weight matrix — pass layer.weight,
                                          not the nn.Linear module itself

    Returns:
        float — total SynOps summed over all timesteps and batch items

    Formula:
        fan_out  = N_post   (each pre neuron connects to all post neurons)
        SynOps   = total_spikes × fan_out
    """
    fan_out      = weight.shape[0]
    total_spikes = spk_rec.sum().item()
    return total_spikes * fan_out


def count_synops_conv(spk_rec: torch.Tensor, conv_layer: nn.Conv2d) -> float:
    """
    Count synaptic operations for a convolutional layer.

    Args:
        spk_rec    : Tensor [T, B, C_in, H, W]
        conv_layer : nn.Conv2d module

    Returns:
        float — total SynOps

    Formula:
        fan_out = C_out × kH × kW
        SynOps  = total_spikes × fan_out
    """
    C_out        = conv_layer.out_channels
    kH, kW       = conv_layer.kernel_size
    fan_out      = C_out * kH * kW
    total_spikes = spk_rec.sum().item()
    return total_spikes * fan_out


def compute_all_synops(spk_recs: dict, model: nn.Module) -> dict:
    """
    Compute SynOps for every layer of the Week 2 SNN architecture.

    Args:
        spk_recs : dict with keys lif1, lif2, lif3, lif4
                   lif1 : [T, B, 32, 28, 28]
                   lif2 : [T, B, 64, 14, 14]
                   lif3 : [T, B, 256]
                   lif4 : [T, B, 10]
        model    : SNN_Week2 instance with .conv2, .fc1, .fc2 attributes

    Returns:
        dict — SynOps per layer plus total
    """
    synops = {}

    # lif1 spikes fan into conv2
    synops["lif1->conv2"] = count_synops_conv(spk_recs["lif1"], model.conv2)

    # lif2 spikes fan into fc1 — flatten spatial dims first
    lif2_flat           = spk_recs["lif2"].flatten(start_dim=2)  # [T, B, 12544]
    synops["lif2->fc1"] = count_synops(lif2_flat, model.fc1.weight)

    # lif3 spikes fan into fc2
    synops["lif3->fc2"] = count_synops(spk_recs["lif3"], model.fc2.weight)

    # lif4 is the output layer — no downstream synaptic cost
    synops["lif4->out"] = 0.0

    synops["total"] = sum(synops.values())
    return synops


def spike_rates(spk_recs: dict) -> dict:
    """
    Compute mean spike rate per layer for monitoring.

    Args:
        spk_recs : dict of spike tensors (any shape)

    Returns:
        dict — mean spike rate per layer key
    """
    return {k: v.mean().item() for k, v in spk_recs.items()}
