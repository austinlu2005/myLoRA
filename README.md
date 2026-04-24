# LoRA Replication

A from-scratch re-implementation of [LoRA: Low-Rank Adaptation of Large Language
Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021), targeting Table 2
of the paper (RoBERTa-base on GLUE).

## Overview

LoRA freezes a pretrained model and injects rank-`r` update matrices
`ΔW = BA` into selected linear projections. For RoBERTa-base on GLUE this means
only the `W_q` and `W_v` attention projections get LoRA adapters, training
~0.3M parameters while matching full fine-tuning accuracy.

This repo contains a clean implementation of the LoRA primitives
(`LoRALinear`, `LoRAConv1D`), the module-replacement injection logic, GLUE
data pipelines, metrics, and a generic training loop — all callable from a
Colab T4 notebook.

## Repository structure

```
project_LoRA/
├── README.md              # this file
├── LICENSE                # MIT
├── .gitignore
├── code/                  # all re-implementation code
│   ├── requirements.txt
│   ├── lora/              # LoRALinear / LoRAConv1D, inject, merge, save/load
│   ├── models/            # architecture wrappers (RoBERTa here)
│   ├── dataloaders/       # GLUE loader (named to avoid colliding with root data/)
│   ├── evaluation/        # GLUE metrics
│   ├── training/          # generic Trainer, optimizer, scheduler
│   ├── utils/             # config, seed, param helpers
│   ├── configs/           # YAML experiment configs
│   ├── scripts/           # CLI entrypoints (train / evaluate / merge)
│   ├── notebook/          # Colab-ready notebook for T4
│   ├── tests/             # unit tests
├── data/                  # README with data acquisition instructions
├── results/               # figures, tables, logs from our runs
├── poster/                # in-class presentation poster (PDF)
└── report/                # final written report (PDF)
```

## Setup

```bash
cd code
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch ≥ 2.0, and `transformers` ≥ 4.40.

## Data

Datasets are fetched on-demand from the Hugging Face Hub — nothing needs to be
committed. See [`data/README.md`](data/README.md) for details and the
one-liner that pre-caches every GLUE task.

## How to run

### Option A — Colab (recommended, T4 GPU)

Open `code/notebook/lora_roberta_glue.ipynb` in Colab. The notebook walks
through uploading the zipped `code/` folder, installing dependencies,
training one or more GLUE tasks, and compiling a Table 2–style results
DataFrame.

### Option B — local / server CLI

```bash
cd code
python scripts/train.py --config configs/roberta_base_mrpc.yaml
```

Swap the config path to run SST-2, CoLA, etc. (see `code/configs/`). Hyperparameters
(rank, α, dropout, learning rate, epochs, max length) all live in the YAML —
nothing is hardcoded in Python.

## Results

Run artifacts (JSON metric histories, plots, merged weights) land in
`results/`. The notebook's final cell produces a per-task comparison against
the paper's reported numbers with a Δ column.

## Authors

Austin Lu — CS 4782, Spring 2026.

## References

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.,
  & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- Reference implementation: <https://github.com/microsoft/LoRA> (used for
  comparison, but not tracked in this repository).

## License

[MIT](LICENSE).
