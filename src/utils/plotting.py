"""
src/utils/plotting.py
Shared plotting utilities for SNN vs ANN analysis figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ── Global style ──────────────────────────────────────────────────────────────

FONT_SIZE   = 13
TITLE_SIZE  = 12
LABEL_SIZE  = 13
TICK_SIZE   = 11
LEGEND_SIZE = 10
FIG_SIZE    = (6.5, 5.0)

SNN_COLOR       = 'steelblue'
SNN_OFF_COLOR   = '#a0b8cc'   # hollow / off-frontier points
ANN_COLOR       = 'crimson'
MARGINAL_COLOR  = 'darkorange'
THRESHOLD_COLOR = 'gray'
SHADE_COLOR     = '#ffe0b2'   # light orange shade for negligible-gain zone


def apply_base_style():
    """Apply shared rcParams. Call once at notebook startup."""
    plt.rcParams.update({
        'font.size'        : FONT_SIZE,
        'font.family'      : 'sans-serif',
        'axes.titlesize'   : TITLE_SIZE,
        'axes.labelsize'   : LABEL_SIZE,
        'xtick.labelsize'  : TICK_SIZE,
        'ytick.labelsize'  : TICK_SIZE,
        'legend.fontsize'  : LEGEND_SIZE,
        'figure.dpi'       : 150,
    })


def pareto_indices(costs, accs):
    """
    Return indices of Pareto-non-dominated points.
    A point is dominated if another point has both lower-or-equal
    energy AND higher-or-equal accuracy, with at least one strict
    inequality.
    """
    pareto = []
    for i, (c, a) in enumerate(zip(costs, accs)):
        dominated = any(
            costs[j] <= c and accs[j] >= a and
            (costs[j] < c or accs[j] > a)
            for j in range(len(costs)) if j != i
        )
        if not dominated:
            pareto.append(i)
    return pareto


def compute_marginal(grouped):
    """
    Compute ΔAcc / Δlog₂T between consecutive T values.
    Returns arrays: marginal_T, marginal_gain, marginal_err.
    Error propagated as sqrt(std_i² + std_{i-1}²).
    """
    g = grouped.sort_values('T').reset_index(drop=True)
    marginal_T    = []
    marginal_gain = []
    marginal_err  = []

    for i in range(1, len(g)):
        delta = g.loc[i, 'acc_mean'] - g.loc[i - 1, 'acc_mean']
        err   = np.sqrt(g.loc[i, 'acc_std'] ** 2 +
                        g.loc[i - 1, 'acc_std'] ** 2)
        marginal_T.append(int(g.loc[i, 'T']))
        marginal_gain.append(delta)
        marginal_err.append(err)

    return (np.array(marginal_T),
            np.array(marginal_gain),
            np.array(marginal_err))


def plot_pareto(grouped, ann_energy_mean, ann_acc_mean, save_path=None):
    """
    Figure 1 — Energy–Accuracy Pareto Frontier.

    Refinements applied:
      A — labels offset individually to avoid overlap in plateau
      B — Pareto dashed line stops at T=8; T=16/32/64 as hollow circles
      C — log-scale x-axis to spread low-T points
      G — font 13–14 pt via apply_base_style()
      H — panel letter 'A' in top-left corner
    """
    apply_base_style()

    costs  = grouped['energy_mean'].values
    accs   = grouped['acc_mean'].values
    T_vals = grouped['T'].values

    pidx       = pareto_indices(costs, accs)
    on_front   = set(pidx)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # ── Off-frontier points (hollow circles) ─────────────────────────────────
    off_idx = [i for i in range(len(grouped)) if i not in on_front]
    if off_idx:
        off = grouped.iloc[off_idx]
        ax.errorbar(
            off['energy_mean'], off['acc_mean'],
            xerr=off['energy_std'], yerr=off['acc_std'],
            fmt='o', color='white', ecolor=SNN_OFF_COLOR,
            elinewidth=1.1, capsize=3, markersize=7,
            markeredgecolor=SNN_OFF_COLOR, markeredgewidth=1.4,
            label='SNN (off frontier)', zorder=3
        )

    # ── On-frontier points (filled circles) ──────────────────────────────────
    on = grouped.iloc[sorted(on_front)]
    ax.errorbar(
        on['energy_mean'], on['acc_mean'],
        xerr=on['energy_std'], yerr=on['acc_std'],
        fmt='o', color=SNN_COLOR, ecolor=SNN_COLOR,
        elinewidth=1.1, capsize=3, markersize=7,
        label='SNN (Pareto frontier)', zorder=4
    )

    # ── Pareto dashed line: stop at T=8 (last meaningful frontier point) ─────
    # T=32 is technically non-dominated but practically meaningless (0.24%
    # gain for 2× energy). Line drawn only through T=1,2,4,8.
    core_front = grouped[
        grouped['T'].isin([1, 2, 4, 8])
    ].sort_values('energy_mean')
    ax.plot(
        core_front['energy_mean'], core_front['acc_mean'],
        '--', color=SNN_COLOR, linewidth=1.4,
        label='Pareto frontier (T=1→8)', zorder=2
    )

    # ── ANN star ─────────────────────────────────────────────────────────────
    ax.scatter(
        ann_energy_mean, ann_acc_mean,
        marker='*', s=240, color=ANN_COLOR, zorder=5,
        label=f'ANN baseline ({ann_acc_mean:.2f}%)'
    )
    ax.annotate(
        'ANN',
        xy=(ann_energy_mean, ann_acc_mean),
        xytext=(7, -14), textcoords='offset points',
        fontsize=10, color=ANN_COLOR
    )

    # ── Per-point T labels with individual offsets to avoid overlap ───────────
    # Offsets tuned for log-scale x-axis
    label_offsets = {
        1:  ( 6,   4),
        2:  ( 6,   4),
        4:  ( 6,  -13),
        8:  ( 6,   4),
        16: ( 6,  -13),
        32: ( 6,   4),
        64: ( 6,  -13),
    }
    for _, row in grouped.iterrows():
        t   = int(row['T'])
        dx, dy = label_offsets.get(t, (6, 4))
        ax.annotate(
            f'T={t}',
            xy=(row['energy_mean'], row['acc_mean']),
            xytext=(dx, dy), textcoords='offset points',
            fontsize=9,
            color=SNN_COLOR if t in [1, 2, 4, 8] else SNN_OFF_COLOR
        )

    # ── Axes — log scale x ────────────────────────────────────────────────────
    ax.set_xscale('log')
    ax.set_xlabel('Estimated Energy (nJ, log scale)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Test Accuracy (%)', fontsize=LABEL_SIZE)

    # ── Panel letter H ────────────────────────────────────────────────────────
    ax.text(0.02, 0.97, 'A', transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')

    ax.set_title(
        'Energy–Accuracy Pareto Frontier\n(Fashion-MNIST, 3 seeds per T)',
        fontsize=TITLE_SIZE
    )
    ax.legend(fontsize=LEGEND_SIZE, loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}")
    plt.show()
    return fig, ax


def plot_marginal(grouped, save_path=None):
    """
    Figure 2 — Marginal Accuracy Gain per Timestep Doubling.

    Refinements applied:
      D — T*=8 annotation arrow correctly positioned
      E — x-axis ticks show T value and log₂T in parentheses
      F — shaded negligible-gain zone below 0.5% threshold
      G — font 13–14 pt via apply_base_style()
      H — panel letter 'B' in top-left corner
    """
    apply_base_style()

    marginal_T, marginal_gain, marginal_err = compute_marginal(grouped)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # ── Shaded negligible-gain zone (below 0.5%) ──────────────────────────────
    ax.axhspan(
        ymin=ax.get_ylim()[0] if ax.get_ylim()[0] < -0.5 else -0.8,
        ymax=0.5,
        color=SHADE_COLOR, alpha=0.35, zorder=0,
        label='Negligible gain zone (< 0.5%)'
    )

    # ── 0.5% threshold line ───────────────────────────────────────────────────
    ax.axhline(0.5, color=THRESHOLD_COLOR, linestyle='--',
               linewidth=1.3, zorder=2,
               label='0.5% threshold (T* criterion)')

    # ── 0% reference line ─────────────────────────────────────────────────────
    ax.axhline(0.0, color='black', linestyle=':', linewidth=0.8, zorder=1)

    # ── Marginal gain curve ───────────────────────────────────────────────────
    ax.errorbar(
        marginal_T, marginal_gain,
        yerr=marginal_err,
        fmt='o-', color=MARGINAL_COLOR, ecolor=MARGINAL_COLOR,
        elinewidth=1.2, capsize=3, markersize=7,
        label='ΔAcc / Δlog₂T', zorder=3
    )

    # ── T*=8 annotation — arrow from label to the T=8 point ──────────────────
    t_star_idx = np.where(marginal_T == 8)[0][0]
    ax.annotate(
        'T* = 8 (elbow)\nlast point > 0.5%',
        xy=(marginal_T[t_star_idx], marginal_gain[t_star_idx]),
        xytext=(12, 35), textcoords='offset points',
        fontsize=9, color=MARGINAL_COLOR,
        arrowprops=dict(
            arrowstyle='->', color=MARGINAL_COLOR,
            lw=1.1,
            connectionstyle='arc3,rad=0.2'
        )
    )

    # ── x-axis: T value + log₂T in parentheses ───────────────────────────────
    log2_labels = {
        2:  '2\n(log₂T=1)',
        4:  '4\n(log₂T=2)',
        8:  '8\n(log₂T=3)',
        16: '16\n(log₂T=4)',
        32: '32\n(log₂T=5)',
        64: '64\n(log₂T=6)',
    }
    ax.set_xscale('log', base=2)
    ax.set_xticks(marginal_T)
    ax.set_xticklabels(
        [log2_labels.get(t, str(t)) for t in marginal_T],
        fontsize=9
    )

    ax.set_xlabel('Timestep T', fontsize=LABEL_SIZE)
    ax.set_ylabel('ΔAccuracy per doubling of T (%)', fontsize=LABEL_SIZE)

    # ── Panel letter H ────────────────────────────────────────────────────────
    ax.text(0.02, 0.97, 'B', transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')

    ax.set_title(
        'Marginal Accuracy Gain per Timestep Doubling\n(Fashion-MNIST, 3 seeds per T)',
        fontsize=TITLE_SIZE
    )
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}")
    plt.show()
    return fig, ax
