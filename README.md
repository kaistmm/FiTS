<div align="center">

# FiTS: Interpretable Spiking Neurons<br>via Frequency Selectivity and Temporal Shaping

[![arXiv](https://img.shields.io/badge/arXiv-2605.13071-b31b1b.svg)](https://arxiv.org/abs/2605.13071)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[**Jongmin Choi**](https://cjongmin.github.io)&nbsp;&nbsp;·&nbsp;&nbsp;**Joon Son Chung**

KAIST

</div>

## News

- 🎉 **2026.09** FiTS has been accepted to NeurIPS as a $`\color{red}\textbf{Spotlight}`$ (0.95%, 292/30,709)!🏅

## Overview

Official implementation of **FiTS**. FiTS learns which frequencies a spiking neuron responds to (Frequency Selectivity, FS) and when those frequencies reach the threshold (Temporal Shaping, TS).

### Frequency Selectivity (FS)

FS uses a LIF neuron with a voltage-dependent adaptation current $a$, with $\mu = 1/\tau_m$ and $\rho = 1/\tau_a$:

$$
\dot V = -\mu V + I - \eta a, \qquad \dot a = -\rho a + \gamma V.
$$

With $\kappa = \eta\gamma$, the subthreshold response $H(j\Omega) = \dfrac{\rho + j\Omega}{(\mu\rho + \kappa - \Omega^2) + j(\mu + \rho)\Omega}$ peaks at the target frequency

$$
\Omega^\star = \sqrt{\sqrt{\kappa(2\rho^2 + 2\rho\mu + \kappa)} - \rho^2},
\qquad
\kappa^\star = \rho(\rho + \mu)\left[\sqrt{1 + \frac{\bigl(1 + (\Omega^\star/\rho)^2\bigr)^2}{(1 + \mu/\rho)^2}} - 1\right].
$$

FiTS learns $f^\star = \Omega^\star / 2\pi$ for each neuron and sets $\kappa$ with the inverse map on the right, so $f^\star$ is what gets initialized, trained and read out. The dynamics are simulated with a semi-implicit Euler step $\Delta t$.

### Temporal Shaping (TS)

TS passes the FS output $V_0$ through first-order all-pass stages

$$
A_m(z) = \frac{z^{-1} - \beta_m}{1 - \beta_m z^{-1}}, \qquad |\beta_m| < 1,
$$

which change the group delay but not the magnitude, and mixes them back with the FS pathway. For one stage,

$$
\tilde V = (1 - \lambda_1) V_0 + \lambda_1 V_1, \qquad \lambda_1 \in [0, 1],
$$

and deeper cascades repeat this mixing stage by stage. Mixing lets the group delay move in either direction, including advances that an all-pass cascade alone cannot produce. $\tilde V$ is the pre-reset voltage; a spike subtracts $V_{\mathrm{th}}$ from the membrane voltage and leaves the adaptation and all-pass states unchanged.

After training, you can read out each neuron's target frequency $f^\star$ and TS group delay $\tau_{\mathrm{TS}}(f^\star)$ ([inspect a trained checkpoint](#inspect-a-trained-checkpoint)). See the [paper](https://arxiv.org/abs/2605.13071) for the derivations.

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

We used Python 3.10, PyTorch 2.5.1 and CUDA 12.1 on an RTX A5000. The installation command above uses CUDA 12.1 wheels; choose a different [PyTorch build](https://pytorch.org/get-started/locally/) if needed for your environment. FiTS uses Triton kernels on CUDA and a slower PyTorch implementation on CPU.

## Quick start

Run a small FiTS network on synthetic spike inputs. No dataset is required.

```bash
python examples/quickstart.py
```

The example runs three optimization steps, then prints the first layer's learned target frequencies, realized discrete-time FS peak frequencies, TS group delays and mixing weights. It selects CUDA when available and otherwise runs on CPU.

## Reproducing the paper

### Prepare the data

Run the commands below from the repository root. Download and decompress SHD or SSC before training; GSC and sMNIST are downloaded automatically on the first run.

| Dataset | Source | `--data-dir` contains |
|---|---|---|
| SHD | [zenkelab.org/datasets](https://zenkelab.org/datasets/) (gunzip the files) | `shd_train.h5`, `shd_test.h5` |
| SSC | [zenkelab.org/datasets](https://zenkelab.org/datasets/) (gunzip the files) | `ssc_train.h5`, `ssc_valid.h5`, `ssc_test.h5` |
| GSC v0.02 | downloaded and split on the first run | `train/`, `valid/`, `test/` |
| sMNIST | downloaded by torchvision | `MNIST/` (created automatically) |

### Train on SHD

After placing `shd_train.h5` and `shd_test.h5` in `data/SHD/`, run:

```bash
python -m SHD.scripts.train.train_chunked \
    --config config/SHD/fits_o1_w128.yaml \
    --data-dir data/SHD
```

This configuration trains for 100 epochs and writes to `runs/shd/fits_o1_w128/`:

- `best.pth`: selected model checkpoint, including its configuration.
- `summary.json`: selected epoch and test accuracy (stored as a fraction).
- `metrics.json` and `train.log`: training history.
- `config.json`: resolved configuration for the run.

Change the output directory with `--results-dir`. A single run takes about 7 minutes on SHD, 24 minutes on GSC and 70 minutes on SSC on an RTX A5000.

### Configurations and reported results

The remaining configurations can be run with:

```bash
python -m SHD.scripts.train.train_chunked --config config/SHD/fits_o2_w256.yaml --data-dir data/SHD
python -m SSC.scripts.train.train_chunked --config config/SSC/fits_o1_w512.yaml --data-dir data/SSC
python -m GSC.scripts.train.train_chunked --config config/GSC/fits_o1_w512.yaml --data-dir data/GSC/processed
python -m MNIST.scripts.train.train_chunked --config config/MNIST/fits_o1_w256.yaml --data-dir data/MNIST
```

| Config | Evaluation protocol | $M$ | Hidden width | Paper accuracy (%) |
|---|---|:-:|:-:|:-:|
| [`SHD/fits_o1_w128`](config/SHD/fits_o1_w128.yaml) | SHD, best test | 1 | 128 | 95.31 ± 0.21 |
| [`SHD/fits_o2_w256`](config/SHD/fits_o2_w256.yaml) | SHD\*, 20% validation split | 2 | 256 | 94.38 ± 0.12 |
| [`SSC/fits_o1_w512`](config/SSC/fits_o1_w512.yaml) | SSC, validation selection | 1 | 512 | 78.23 ± 0.16 |
| [`GSC/fits_o1_w512`](config/GSC/fits_o1_w512.yaml) | GSC, validation selection | 1 | 512 | 94.48 ± 0.12 |
| [`MNIST/fits_o1_w256`](config/MNIST/fits_o1_w256.yaml) | sMNIST, best test | 1 | 256 | 98.28 |

$M$ is the number of TS all-pass stages (`--order`); $M=0$ disables TS. Hidden width is the width of each hidden layer.

**Checkpoint selection:** SHD and sMNIST report the highest test accuracy across epochs. SHD\* holds out 20% of the training set and selects the epoch by validation accuracy. SSC and GSC also select by validation accuracy and report the test accuracy of the selected checkpoint.

The table contains the paper's reported results. All training configurations and the built-in training defaults use `seed: 0`. Override it with `--seed` when needed.

Training options can be overridden on the command line. For example, append `--order 0` for FS only or `--neuron-type lif` for the LIF baseline. Use `python -m SHD.scripts.train.train_chunked --help` to see the available flags.

### Inspect a trained checkpoint

```bash
python examples/inspect_checkpoint.py runs/shd/fits_o1_w128/best.pth \
    --csv neurons.csv --plot target_frequencies.png
```

This prints per-layer frequency and delay statistics, exports one CSV row per neuron, and plots the learned target-frequency distributions. The readouts distinguish the learned target $f^\star$ from the realized discrete-time FS peak $f^\star_{\mathrm{DT}}$; these can differ under the default continuous-time parameterization.

## Using FiTS in your own model

```python
import torch
from fits import FiTSNeuron, FrequencyGuard

device = "cuda" if torch.cuda.is_available() else "cpu"
layer = FiTSNeuron(
    in_features=140, out_features=256, order=1,
    tau_m=0.04, tau_a=0.2, dt=0.004, f_min=1.0, f_max=50.0,
).to(device)
x = (torch.rand(8, 250, 140, device=device) < 0.05).float()
spikes, *_ = layer(x)                          # [batch, time, 256]

optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
guard = FrequencyGuard(layer, optimizer)       # applied on optimizer steps

target_hz = layer.get_freq()
realized_hz = layer.realized_freq()
delay_s = layer.ts_group_delay()
```

Inputs have shape `[batch, time, in_features]`. `FiTSNeuron` includes a learned linear projection followed by the spiking dynamics. It returns spikes and the final membrane, adaptation and all-pass states; the example above keeps only the spikes.

`dt` is the input time step $\Delta t$ in seconds. `f_min` and `f_max` specify the initial range of $f^\star$ in Hz, rather than bounds enforced throughout training.

The TS readout is the group delay of the mixed all-pass cascade $G$, evaluated at each neuron's target frequency:

$$
\tau_{\mathrm{TS}}(f^\star)
= -\Delta t\left.
\frac{\mathrm{d}}{\mathrm{d}\omega}\arg G(e^{j\omega})
\right|_{\omega = 2\pi f^\star\Delta t}.
$$

Here $\omega$ is angular frequency in radians per sample, so the returned delay is in seconds. Negative values indicate a group-delay advance.

<details>
<summary>Frequency stability during training</summary>

The benchmark scripts enable `FrequencyGuard` by default. It keeps the learned $f^\star$ below a safety cap derived from the stability limit of the semi-implicit FS update. The stability limits before applying the safety margin are about 30.9 Hz for the GSC constants and 77.2 Hz for SHD/SSC.

The guard runs through optimizer hooks and leaves the forward computation unchanged. To disable it in a benchmark run, pass `--freq-guard false`.

</details>

## Code structure

```text
SHD/                           # SSC/, GSC/ and MNIST/ follow the same layout
├── core/                      # data loading and training utilities
├── models/fc.py                # dataset-specific network construction
├── scripts/train/train_chunked.py
└── spiking_neuron/             # FiTS/LIF exports from the shared implementation
fits/                          # shared neuron dynamics, models and Triton kernels
common/                        # shared training loop and SHD/SSC event loading
config/{SHD,SSC,GSC,MNIST}/      # paper experiment configurations
examples/                      # quickstart and checkpoint inspection
tests/test_fits.py              # numerical and gradient checks for the implementation
```

The dataset layout follows the anonymous release. Shared neuron code stays in `fits/` so fixes apply consistently to every dataset. `tests/test_fits.py` checks the implementation, including agreement between Triton and PyTorch; it is not a benchmark evaluation script and is not required for training. Developers can run it with `python -m pytest tests`.

## Citation

```bibtex
@inproceedings{choi2026fits,
  title     = {{FiTS}: Interpretable Spiking Neurons via Frequency Selectivity and Temporal Shaping},
  author    = {Choi, Jongmin and Chung, Joon Son},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

Apache-2.0, see [LICENSE](LICENSE). The datasets keep their own licenses.
