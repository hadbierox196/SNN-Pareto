# SNN-Pareto

[![DOI](https://zenodo.org/badge/1268503330.svg)](https://doi.org/10.5281/zenodo.20726698)

**Companion code for:**
Farooq, H. (2025). Energy--Accuracy Trade-offs in Spiking Neural Networks: A Pareto Analysis on Fashion-MNIST.
DOI: https://doi.org/10.5281/zenodo.20726698

---

## Overview

This repository contains all training code, synaptic operations (SynOps) counting implementations, energy estimation scripts, model checkpoints, and analysis notebooks associated with the above manuscript. The study characterises the energy--accuracy Pareto frontier of a directly-trained, rate-coded spiking neural network (SNN) across seven timestep budgets (*T* in {1, 2, 4, 8, 16, 32, 64}) on Fashion-MNIST, benchmarked against full ANN and capacity-constrained SmallANN baselines under the Lemaire et al. (2022) analytical energy model at 45 nm CMOS.

The central finding is that *T* = 8 constitutes the practical plateau onset: accuracy gains beyond this point are statistically indistinguishable across *T* = 8--64 (spanning 0.23%), while energy costs scale by up to 5.68x. At *T* = 64, cumulative synaptic operation volume causes the SNN to exceed the full ANN's energy consumption (22,228.6 nJ vs. 21,361.7 nJ) while remaining 3.31% less accurate. All energy figures are theoretical lower bounds under the Horowitz 45 nm CMOS constants and should not be interpreted as absolute hardware measurements.

---

## Repository Structure

```
SNN-Pareto/
|
|-- src/                  # Model definitions, training loops, SynOps counter, energy model, plotting utilities
|-- notebooks/            # One notebook per experimental stage
|-- checkpoints/          # Saved model weights (.pt), one per seed per configuration
|-- results/              # Per-seed CSV outputs (accuracy, SynOps, energy)
|-- figures/              # Generated figures including the Pareto frontier plot
|-- docs/                 # Validation logs, literature notes, review documents
|
|-- reproduce.py          # Runs the full pipeline end-to-end
|-- requirements.txt      # All dependencies pinned
|-- README.md
```

---

## Models

**Full ANN.** Two convolutional blocks (Conv2d 1->32 and 32->64, 3x3 kernels, padding 1, MaxPool2d 2x2) followed by two fully-connected layers (3136->256 with Dropout 0.3, 256->10). Total parameters: 824,458. Estimated energy: 21,361.7 nJ per inference (4,643,840 MACs x 4.6 pJ).

**SNN.** Identical weight dimensions to the Full ANN. Each ReLU replaced by a Leaky Integrate-and-Fire (LIF) neuron with membrane decay beta = 0.9 and threshold = 1.0. Surrogate gradient: fast sigmoid, slope = 25 (Zenke & Ganguli, 2018; Neftci et al., 2019). Inputs are Poisson rate-encoded over *T* timesteps; classification is based on summed output-layer spike count across *T*. This is a direct-training architecture implemented in snnTorch -- not an ANN-to-SNN conversion. Convolutional weights are initialised from a matched-seed ANN checkpoint; fully-connected layers are randomly re-initialised.

**SmallANN.** Capacity-constrained ANN ablation with channels halved at every layer (Conv2d 1->16, 16->32; Linear 1568->64, 64->10). Total parameters: 105,866 (7.8x fewer than the Full ANN). Confirms that the SNN energy advantage at *T* = 4--8 cannot be attributed to parameter-count reduction alone.

---

## Training Configuration

All models were trained under the following fixed configuration:

- Optimiser: Adam, learning rate 1e-3
- Scheduler: ReduceLROnPlateau (mode = max, patience = 3, factor = 0.5)
- Batch size: 128
- Epochs: 20
- Gradient clipping: max_norm = 1.0
- Input normalisation: mean = 0.2860, std = 0.3530
- Validation split: 10% stratified random hold-out (6,000 images)
- Checkpoint selection: epoch with highest validation accuracy
- Seeds: 42, 123, 7 (three independent runs per configuration)

A total of 21 independent SNN models were trained (7 timestep values x 3 seeds), plus 6 ANN/SmallANN models (2 architectures x 3 seeds). All SNN models at each *T* were trained from scratch.

---

## Energy Model

Energy per inference is estimated using the Lemaire et al. (2022) analytical framework with Horowitz (2014) operation costs at 45 nm CMOS:

- SNN: E = SynOps x 0.9 pJ (accumulate-only; pre-synaptic input is binary, no multiply required)
- ANN: E = MACs x 4.6 pJ (multiply-accumulate)

SynOps are counted on the complete 10,000-image test set and reported as per-sample means. The FC formula (sum over timesteps, batch, and pre-synaptic neurons of s_i(t,b) x N_post) and the convolutional formula (total_spikes x C_out x kH x kW) were each validated against hand-calculated toy networks prior to use. A sensitivity check across AC/MAC ratios of 2x to 6x confirms that the Pareto non-dominance of *T* in {1, 2, 4, 8} and the *T* = 64 energy inversion are robust to this variation.

All energy figures exclude DRAM access costs, membrane potential state memory, and spike-routing overhead. They are comparative proxies and should not be interpreted as predictions for physical neuromorphic hardware.

---

## Key Results

| T        | Accuracy (mean +/- std) | Energy (nJ) |
|----------|-------------------------|-------------|
| 1        | 80.67% +/- 0.67%        | 697.9       |
| 2        | 84.85% +/- 0.30%        | 2,040.9     |
| 4        | 88.30% +/- 0.18%        | 3,375.8     |
| 8        | 89.69% +/- 0.14%        | 3,910.1     |
| 16       | 89.54% +/- 0.33%        | 6,337.7     |
| 32       | 89.77% +/- 0.27%        | 12,372.6    |
| 64       | 89.66% +/- 0.39%        | 22,228.6    |
| Full ANN | 92.97%                  | 21,361.7    |
| SmallANN | 91.61% +/- 0.07%        | 5,138.5     |

Pareto-non-dominated SNN operating points: *T* in {1, 2, 4, 8}. No SNN operating point dominates the Full ANN. 95% bootstrap confidence intervals (1,000 resamples, percentile method, seed = 42) are reported in full in Table 2 of the manuscript. With n = 3 seeds, CIs are bounded by the observed data range and are reported as indicators of seed-level variability rather than as intervals carrying conventional 95% coverage guarantees.

---

## Environment

| Library     | Version       |
|-------------|---------------|
| PyTorch     | 2.11.0+cu128  |
| snnTorch    | 0.9.4         |
| torchvision | 0.26.0+cu128  |
| NumPy       | 2.0.2         |
| Matplotlib  | 3.10.0        |
| SciPy       | >= 1.11.0     |
| pandas      | >= 2.0.0      |
| tqdm        | >= 4.65.0     |
| CUDA Device | Tesla T4      |

Install dependencies:

```
pip install -r requirements.txt
```

---

## Reproducing the Experiments

To run the full pipeline end-to-end:

```
git clone https://github.com/hadbierox196/SNN-Pareto.git
cd SNN-Pareto
pip install -r requirements.txt
python reproduce.py
```

Individual notebooks in `notebooks/` correspond to each experimental stage (ANN baseline, SNN sweep, SynOps counting, Pareto analysis, ablation). Refer to the inline documentation within each notebook for stage-specific instructions.

---

## Data

Fashion-MNIST is downloaded automatically via `torchvision.datasets.FashionMNIST` on first run. The dataset is publicly available at https://github.com/zalandoresearch/fashion-mnist (Xiao et al., 2017). No additional data acquisition is required. Underlying per-seed CSV result files are included in `results/` and are available from the corresponding author upon reasonable request.

---

## Citation

If you use this code or the associated results, please cite the manuscript and this repository:

```
Farooq, H. (2025). Energy--Accuracy Trade-offs in Spiking Neural Networks:
A Pareto Analysis on Fashion-MNIST. 
```

A DOI-archived snapshot of this repository will be deposited on Zenodo to ensure long-term reproducibility.

---

## License

MIT License. See `LICENSE` for terms.

---

## Author

Hassan Farooq  
Sargodha Medical College  
hasanfarooq.edu@gmail.com  
ORCID: 0009-0000-1269-3885
