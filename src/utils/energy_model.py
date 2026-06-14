"""
src/utils/energy_model.py

Energy estimation for ANN vs SNN comparison.
Implements Lemaire et al. (2022) 45nm CMOS model.

Constants:
    E_AC  = 0.9e-12 J  per accumulate          (SNN spike-driven op)
    E_MAC = 4.6e-12 J  per multiply-accumulate  (ANN dense op)

Reference:
    Lemaire et al. (2022) "An Analytical Estimation of Spiking Neural
    Networks Energy Efficiency." https://arxiv.org/abs/2209.10074
"""

# ── Lemaire et al. 45nm CMOS constants ────────────────────────────────
E_AC  = 0.9e-12   # joules — accumulate only (SNN, binary spike input)
E_MAC = 4.6e-12   # joules — multiply-accumulate (ANN, real-valued input)


def energy_snn(total_synops: float) -> float:
    """
    Compute SNN inference energy.

    Each SynOps is one AC (accumulate) operation — no multiply
    because the pre-synaptic signal is binary (0 or 1).

    Args:
        total_synops : float — total synaptic operations

    Returns:
        float — energy in joules
    """
    return total_synops * E_AC


def energy_ann(total_macs: float) -> float:
    """
    Compute ANN inference energy.

    Each operation is one MAC (multiply-accumulate) — real-valued
    activations require a multiply at every synapse.

    Args:
        total_macs : float — total multiply-accumulate operations

    Returns:
        float — energy in joules
    """
    return total_macs * E_MAC


def count_ann_macs_week2() -> dict:
    """
    Analytical MAC count for the Week 2 architecture (per sample).

    Architecture:
        conv1 : Conv2d(1->32,  3x3, pad=1), input 28x28
        conv2 : Conv2d(32->64, 3x3, pad=1), input 14x14 (after pool1)
        fc1   : Linear(3136->256)
        fc2   : Linear(256->10)

    Formula per layer:
        Conv : C_out x C_in x kH x kW x H_out x W_out
        FC   : N_in  x N_out

    Returns:
        dict with per-layer MACs and total
    """
    conv1_macs = 32  * 1  * 3 * 3 * 28 * 28   # 225,792
    conv2_macs = 64  * 32 * 3 * 3 * 14 * 14   # 3,612,672
    fc1_macs   = 3136 * 256                    # 802,816
    fc2_macs   = 256  * 10                     # 2,560
    total      = conv1_macs + conv2_macs + fc1_macs + fc2_macs  # 4,643,840
    return {
        "conv1": conv1_macs,
        "conv2": conv2_macs,
        "fc1"  : fc1_macs,
        "fc2"  : fc2_macs,
        "total": total,
    }


def energy_comparison_table(ann_macs: float,
                             snn_synops_per_sample: float,
                             T: int = 32) -> dict:
    """
    Compute and return the full ANN vs SNN energy comparison.

    Args:
        ann_macs               : total ANN MACs per sample
        snn_synops_per_sample  : total SNN SynOps per sample
        T                      : timesteps used (for labelling only)

    Returns:
        dict with keys:
            ann_ops, snn_ops,
            ann_energy_J, snn_energy_J,
            ann_energy_nJ, snn_energy_nJ,
            efficiency_x, T
    """
    ann_energy_J  = energy_ann(ann_macs)
    snn_energy_J  = energy_snn(snn_synops_per_sample)
    ann_energy_nJ = ann_energy_J * 1e9
    snn_energy_nJ = snn_energy_J * 1e9
    efficiency    = ann_energy_nJ / snn_energy_nJ if snn_energy_nJ > 0 else float("inf")

    return {
        "T"             : T,
        "ann_ops"       : ann_macs,
        "snn_ops"       : snn_synops_per_sample,
        "ann_energy_J"  : ann_energy_J,
        "snn_energy_J"  : snn_energy_J,
        "ann_energy_nJ" : ann_energy_nJ,
        "snn_energy_nJ" : snn_energy_nJ,
        "efficiency_x"  : efficiency,
    }


def print_energy_table(result: dict) -> None:
    """Pretty-print the output of energy_comparison_table()."""
    T = result["T"]
    print("═" * 65)
    print(f"{'ENERGY COMPARISON (per sample, T=' + str(T) + ')':^65}")
    print("═" * 65)
    print(f"  {'Metric':<35} {'ANN':>12}  {'SNN':>12}")
    print("─" * 65)
    print(f"  {'Operations (MACs / SynOps)':<35} "
          f"{result['ann_ops']:>12,.0f}  {result['snn_ops']:>12,.0f}")
    print(f"  {'Energy (nJ)':<35} "
          f"{result['ann_energy_nJ']:>12.4f}  {result['snn_energy_nJ']:>12.4f}")
    print(f"  {'SNN saving vs ANN':<35} "
          f"{'':>12}  {result['efficiency_x']:>11.2f}x")
    print("═" * 65)
    print()
    print("  Lemaire constants: E_AC = 0.9 pJ | E_MAC = 4.6 pJ | 45nm node")
