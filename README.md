<div align="center">

# FiTS: Interpretable Spiking Neurons<br>via Frequency Selectivity and Temporal Shaping

[![arXiv](https://img.shields.io/badge/arXiv-2605.13071-b31b1b.svg)](https://arxiv.org/abs/2605.13071)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[**Jongmin Choi**](https://cjongmin.github.io)&nbsp;&nbsp;·&nbsp;&nbsp;**Joon Son Chung**

KAIST

</div>

## News

- 🎉 **2026.09** FiTS has been accepted to $`\color{red}\text{𝗡𝗲𝘂𝗿𝗜𝗣𝗦}`$ as a $`\color{red}\text{𝗦𝗽𝗼𝘁𝗹𝗶𝗴𝗵𝘁}`$ ($`\color{red}\text{𝟬.𝟵𝟱\%}`$)!🏅

## Overview

Official implementation of **FiTS**.

Our spiking neuron, FiTS, consists of two modules:

- **Frequency Selectivity (FS)**: learns which temporal frequencies each neuron emphasizes, through an explicit target frequency $f^\star$.
- **Temporal Shaping (TS)**: reshapes when those frequency components contribute to pre-spike membrane voltage accumulation, through group-delay modulation.

## Method

### Frequency Selectivity (FS)

FS uses a LIF neuron with a voltage-dependent adaptation current $a$, with $\mu = 1/\tau_m$ and $\rho = 1/\tau_a$:

$$
\dot V = -\mu V + I - \eta a, \qquad \dot a = -\rho a + \gamma V.
$$

With $\kappa = \eta\gamma$, the continuous-time subthreshold frequency response is

$$
H(j\Omega)
= \frac{V(j\Omega)}{I(j\Omega)}
= \frac{\rho + j\Omega}
{(\mu\rho + \kappa - \Omega^2) + j(\mu + \rho)\Omega}.
$$

When $|H(j\Omega)|$ has a unique nonzero global maximizer over $\Omega > 0$, the corresponding target angular frequency and its inverse map are

$$
\Omega^\star = \sqrt{\sqrt{\kappa(2\rho^2 + 2\rho\mu + \kappa)} - \rho^2},
\qquad
\kappa^\star = \rho(\rho + \mu)\left[\sqrt{1 + \frac{\bigl(1 + (\Omega^\star/\rho)^2\bigr)^2}{(1 + \mu/\rho)^2}} - 1\right].
$$

FiTS uses $f^\star_{\mathrm{CT}} = \Omega^\star / 2\pi$ as each neuron's trainable coordinate and computes $\kappa^\star$ with the inverse map on the right. Thus, $f^\star_{\mathrm{CT}}$ is initialized, optimized and interpreted directly. The dynamics are simulated with a semi-implicit Euler step $\Delta t$.

### Temporal Shaping (TS)

TS passes the FS output $V_0$ through an $M$-stage cascade of first-order all-pass filters

$$
A_m(z) = \frac{z^{-1} - \beta_m}{1 - \beta_m z^{-1}}, \qquad |\beta_m| < 1,
$$

which preserve magnitude while modifying phase, and recursively mixes the stage outputs with the direct FS pathway. For one stage,

$$
\tilde V = (1 - \lambda_1) V_0 + \lambda_1 V_1, \qquad \lambda_1 \in [0, 1],
$$

and deeper cascades repeat this mixing stage by stage. The TS-induced group-delay shift can be positive or negative, whereas a pure all-pass cascade contributes only nonnegative group delay. The final mixed output $\tilde V_M$ is the effective pre-reset voltage; a spike subtracts $V_{\mathrm{th}}$ from the membrane voltage and leaves the adaptation and all-pass states unchanged.

After training, you can read out each neuron's continuous-time target frequency $f^\star_{\mathrm{CT}}$ and TS-induced group-delay shift $\Delta\tau(f^\star_{\mathrm{CT}})$ ([evaluate a trained checkpoint](#evaluation)). See the [paper](https://arxiv.org/abs/2605.13071) for the derivations.

## Installation

```bash
git clone https://github.com/kaistmm/FiTS.git
cd FiTS
conda create -n fits python=3.10 -y
conda activate fits
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

## Data

Download and decompress SHD or SSC before training. GSC and sMNIST are downloaded automatically on the first run.

| Dataset | Source | `--data-dir` contains |
|---|---|---|
| SHD | [zenkelab.org/datasets](https://zenkelab.org/datasets/) (gunzip the files) | `shd_train.h5`, `shd_test.h5` |
| SSC | [zenkelab.org/datasets](https://zenkelab.org/datasets/) (gunzip the files) | `ssc_train.h5`, `ssc_valid.h5`, `ssc_test.h5` |
| GSC v0.02 | downloaded and split on the first run | `train/`, `valid/`, `test/` |
| sMNIST | downloaded by torchvision | `MNIST/` (created automatically) |

## Train

Run the commands from the repository root. Each command launches one training run using the architecture, neuron constants and optimization settings in its YAML file.

```bash
# SHD
python -m SHD.scripts.train.train_chunked \
    --config config/SHD/fits_o1_w128.yaml \
    --data-dir data/SHD

# SHD with a 20% validation split
python -m SHD.scripts.train.train_chunked \
    --config config/SHD/fits_o2_w256.yaml \
    --data-dir data/SHD

# SSC
python -m SSC.scripts.train.train_chunked \
    --config config/SSC/fits_o1_w512.yaml \
    --data-dir data/SSC

# GSC
python -m GSC.scripts.train.train_chunked \
    --config config/GSC/fits_o1_w512.yaml \
    --data-dir data/GSC/processed

# sMNIST
python -m MNIST.scripts.train.train_chunked \
    --config config/MNIST/fits_o1_w256.yaml \
    --data-dir data/MNIST
```

The GSC and sMNIST commands prepare their datasets automatically. The SHD and SSC commands expect the files listed in [Data](#data).

Training options can be overridden on the command line. For example, append `--order 0` for FS only, `--neuron-type lif` for the LIF baseline, `--seed <int>` to choose a random seed, or `--results-dir <path>` to change the output directory. Use any entry point with `--help` to list all options.

$M$ is the number of TS all-pass stages (`--order`); $M=0$ disables TS.

## Inference

A `FiTSNeuron` can be used directly as a sequence layer:

```python
import torch
from fits import FiTSNeuron

device = "cuda" if torch.cuda.is_available() else "cpu"
layer = FiTSNeuron(
    in_features=140, out_features=256, order=1,
    tau_m=0.04, tau_a=0.2, dt=0.004, f_min=1.0, f_max=50.0,
).to(device)
x = (torch.rand(8, 250, 140, device=device) < 0.05).float()
with torch.no_grad():
    spikes, *_ = layer(x)                      # [batch, time, 256]
```

Inputs have shape `[batch, time, in_features]`. `FiTSNeuron` includes a learned linear projection followed by the spiking dynamics. It returns spikes and the final membrane, adaptation and all-pass states; the example above keeps only the spikes.

`dt` is the input time step $\Delta t$ in seconds. `f_min` and `f_max` specify the initial range of $f^\star_{\mathrm{CT}}$ in Hz, rather than bounds enforced throughout training.

## Evaluation

Each run writes the following files under the configured `results_dir`:

- `best.pth`: model checkpoint and resolved configuration.
- `summary.json`: test accuracy.
- `metrics.json` and `train.log`: training history.
- `config.json`: resolved run configuration.

Inspect the learned neuron parameters in a checkpoint with:

```bash
python examples/inspect_checkpoint.py runs/shd/fits_o1_w128/best.pth \
    --csv neurons.csv --plot target_frequencies.png
```

The command prints per-layer statistics, exports one CSV row per neuron and plots the learned target-frequency distributions. The readouts distinguish the learned continuous-time target $f^\star_{\mathrm{CT}}$ from the realized discrete-time target $f^\star_{\mathrm{DT}}$.

The same readouts are available directly from a `FiTSNeuron`:

```python
target_hz = layer.get_freq()
realized_hz = layer.realized_freq()
delay_s = layer.ts_group_delay()
```

`ts_group_delay()` returns the group-delay shift induced by the mixed response $G$, evaluated at each neuron's continuous-time target frequency:

$$
\Delta\tau(f^\star_{\mathrm{CT}})
= -\Delta t\left.
\frac{\mathrm{d}}{\mathrm{d}\omega}\arg G(e^{j\omega})
\right|_{\omega = 2\pi f^\star_{\mathrm{CT}}\Delta t}.
$$

Here $G$ is the TS module's mixed response relative to the direct FS pathway, and $\omega$ is angular frequency in radians per sample. The returned shift is in seconds; negative values indicate a group-delay advance.


## Code structure
```text
SHD/                           # SSC/, GSC/ and MNIST/ follow the same layout
├── core/                      # data loading and training utilities
├── models/fc.py                # dataset-specific network construction
├── scripts/train/train_chunked.py
└── spiking_neuron/             # FiTS/LIF exports from the shared implementation
fits/                          # shared neuron dynamics, models and Triton kernels
common/                        # shared training loop and SHD/SSC event loading
config/{SHD,SSC,GSC,MNIST}/      # provided training configurations
examples/inspect_checkpoint.py   # checkpoint-level neuron readouts
tests/test_fits.py              # numerical and gradient checks for the implementation
```

Shared neuron code stays in `fits/` so fixes apply consistently to every dataset. `tests/test_fits.py` checks the implementation, including agreement between Triton and PyTorch. Developers can run it with `python -m pytest tests`.

## Citation

```bibtex
@inproceedings{choi2026fits,
  title     = {{FiTS}: Interpretable Spiking Neurons via Frequency Selectivity and Temporal Shaping},
  author    = {Choi, Jongmin and Chung, Joon Son},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026}
}
```

Licensed under [Apache-2.0](LICENSE). The datasets keep their own licenses.
