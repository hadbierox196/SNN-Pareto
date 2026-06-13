# snn-pareto

> Marginal Energy Cost of Inference Timesteps in Spiking vs. Standard Neural Networks

**Status:** Week 1 Complete — ANN Baseline Locked  
**Dataset:** Fashion-MNIST  
**arXiv:** _coming Week 13_

---

## What This Project Does

Measures the marginal energy cost of each additional inference timestep
in spiking neural networks (SNNs) versus standard ANNs, using the
Lemaire et al. (2022) energy model on a shared convolutional backbone.
Produces a Pareto frontier (energy vs. accuracy) and a marginal gain
curve (ΔAccuracy / Δlog₂T) to identify the optimal operating timestep T*.

---

## Repository Structure
snn-pareto/

├── notebooks/        # One notebook per week (01 through 07)

├── src/

│   ├── models/       # ANN, SNN, SmallANN class definitions

│   ├── training/     # Training loops and sweep runner

│   └── utils/        # SynOps counter, energy model, plotting

├── results/          # CSVs from all experiments

├── figures/          # All plots and publication-quality figures

├── paper/            # Manuscript drafts (added Week 9+)

├── docs/             # Validation logs, literature notes, review docs

├── reproduce.py      # Runs full pipeline end-to-end

└── requirements.txt  # All dependencies pinned


---

## Reproduce Results

```bash
git clone https://github.com/hadbierox196/snn-pareto.git
cd snn-pareto
pip install -r requirements.txt
python reproduce.py
```

Expected runtime: ~X hours on Colab T4 _(fill in after Week 7)_

---

## Results Summary

| Model | Test Accuracy | Energy (nJ) |
|---|---|---|
| ANN Baseline (824K params) | 92.05% | ___ |
| SNN T=32 | ___ % | ___ |
| SNN T* (elbow point) | ___ % | ___ |
| Small ANN ~46K params | ___ % | ___ |

_to be Fill in after Week 5 and Week 6._

---

## Environment

| Library | Version |
|---|---|
| PyTorch | 2.11.0+cu128 |
| snnTorch | 0.9.4 |
| Torchvision | 0.26.0+cu128 |
| NumPy | 2.0.2 |
| Matplotlib | 3.10.0 |
| CUDA Device | Tesla T4 |

---

## Weekly Progress Log

| Week | Goal | Status |
|---|---|---|
| 1 | ANN Baseline ≥92% | ✅ 92.05% |
| 2 | SNN Convergence >80% at T=32 | ⬜ |
| 3 | SynOps Counter Verified | ⬜ |
| 4 | Full Timestep Sweep (21 runs) | ⬜ |
| 5 | Pareto + Marginal Figures | ⬜ |
| 6 | Iso-Parameter Ablation | ⬜ |
| 7 | Bootstrap CIs + reproduce.py | ⬜ |
| 8 | Literature Review | ⬜ |
| 9 | Methods + Results Draft | ⬜ |
| 10 | Full Paper Draft | ⬜ |
| 11 | Self-Review Pass | ⬜ |
| 12 | Revision + Polish | ⬜ |
| 13 | Journal Submission | ⬜ |
| 14 | Post-Submission | ⬜ |

---

## License

MIT License — see `LICENSE` for details.

## Author

Independent Researcher  
ORCID: 0009-0000-1269-3885
