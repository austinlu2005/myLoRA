# Re-implementation of LoRA: Low-Rank Adaptation of Large Language Models

CS 4782 (Cornell, Spring 2026) final project — a from-scratch re-implementation
of [Hu et al., 2021](https://arxiv.org/abs/2106.09685).

## 1. Introduction

This repository reproduces **LoRA: Low-Rank Adaptation of Large Language
Models** (Hu et al., 2021). The paper's central contribution is a
parameter-efficient fine-tuning method that freezes a pretrained model and
injects trainable rank-`r` update matrices `ΔW = BA` into selected linear
projections — matching full fine-tuning quality while training **<1%** of the
parameters and adding **zero** inference latency once `BA` is folded back into
the base weight.

## 2. Chosen Result

We target **Table 2** of the paper: RoBERTa-base + LoRA on the eight GLUE
tasks. We additionally reproduce the GPT-2 Medium row of **Table 3** (E2E NLG
generation) and extend the same `LoRALinear` primitive to a vision-language
model (CLIP) as our novel contribution.

> "LoRA can match or exceed the fine-tuning baseline […] while having only
> 0.27M trainable parameters" — Hu et al., 2021, Table 2.

## 3. GitHub Contents

```
myLoRA/
├── code/        re-implementation source (lora primitives, model wrappers,
│                trainers, configs, scripts, notebooks, unit tests)
├── data/        no raw data — README explains how to fetch each dataset
├── results/     per-task JSON metric histories, headline comparison CSV,
│                E2E test-set scores, raw run zips
├── poster/      in-class presentation poster (PDF + source)
├── report/      final written report (PDF)
├── LICENSE      MIT
├── .gitignore
└── README.md    this file
```

## 4. Re-implementation Details

- **Core LoRA primitive** (`code/lora/layers.py`) — `LoRALinear` wraps any
  `nn.Linear`, freezes its weight, adds two trainable matrices `A` (Kaiming
  init) and `B` (zero init) so the update starts at exactly `0`, scales by
  `α/r`, and exposes `merge`/`unmerge` for zero-overhead inference.
- **GPT-2 specifics** — HuggingFace's `Conv1D` stores weight transposed;
  `LoRAConv1D` handles that. The fused `c_attn` QKV projection is adapted by
  `LoRAConv1DQV`, which keeps Q and V independent and leaves K untouched (paper
  prescription).
- **Injection** (`code/lora/inject.py`) — walks `named_modules()`, swaps every
  matched module on its parent via `setattr`, and freezes everything that
  isn't a LoRA parameter. The original linear lives on as `base_layer` inside
  the wrapper.
- **Models / datasets / metrics** — RoBERTa-base on all 8 GLUE tasks; GPT-2
  Medium on E2E NLG (causal LM with prompt masking); CLIP-ViT-Base/32 on
  Pokémon image-caption retrieval. GLUE metrics via the `evaluate` library;
  E2E via the official `e2e-metrics` scorer; CLIP via image↔text Recall@K.
- **Hyperparameters** copy the paper's Table 11/12 exactly: rank 8 (RoBERTa) /
  rank 4 (GPT-2), AdamW with linear warmup, label smoothing 0.1 for E2E.
- **Modifications** — none to the LoRA mechanism itself. The novel piece is
  applying the same module-replacement injector to a multi-tower CLIP model
  (`code/models/vlm_wrapper.py`) using prefix-filtered targeting.

## 5. Reproduction Steps

**Environment.** Python 3.10, PyTorch ≥ 2.0, `transformers ≥ 4.40`. A single
T4 / L4 GPU is sufficient for every experiment in this repo.

```bash
git clone <this-repo>
cd myLoRA/code
pip install -r requirements.txt
```

**RoBERTa GLUE replication** (Table 2):
```bash
python scripts/train.py --config configs/roberta_base_mrpc.yaml
# swap the YAML for sst2, cola, mnli, qqp, qnli, rte, stsb
```

**GPT-2 Medium E2E NLG** (Table 3):
```bash
python scripts/train.py --config configs/gpt2_medium_e2e.yaml
python scripts/evaluate_e2e.py --lora-path runs/gpt2_medium_e2e/lora_best.pt
```

**CLIP vision-language extension:**
```bash
python scripts/train_vlm_lora_clip.py --config configs/vlm_clip_pokemon.yaml
```

**Notebooks** (`code/notebook/`) wrap each experiment for one-click Colab
runs: `lora_roberta_glue.ipynb`, `lora_gpt2_medium.ipynb`,
`eval_e2e_lora.ipynb`, `lora_clip_vlm_retrieval.ipynb`.

## 6. Results / Insights

End-to-end GLUE + E2E comparison vs. paper (full table in
`results/comparison_glue_e2e.csv`):

| Benchmark | Task  | Metric             | Paper | Ours  | Δ     |
|-----------|-------|--------------------|------:|------:|------:|
| GLUE      | MNLI  | accuracy           | 87.5  | **87.9** | +0.4 |
| GLUE      | SST-2 | accuracy           | 95.1  | 94.8  | −0.3  |
| GLUE      | QQP   | accuracy           | 90.8  | **90.8** | 0.0  |
| GLUE      | QNLI  | accuracy           | 93.3  | 92.8  | −0.5  |
| GLUE      | MRPC  | accuracy           | 89.7  | 88.2  | −1.5  |
| GLUE      | CoLA  | Matthews corr.     | 63.4  | 62.3  | −1.1  |
| GLUE      | STS-B | Pearson corr.      | 91.5  | 90.9  | −0.6  |
| GLUE      | RTE   | accuracy           | 86.6  | 79.1  | −7.5  |
| E2E       | NLG   | BLEU               | 70.4  | 69.8  | −0.6  |

Match or beat paper on the data-rich tasks (MNLI, QQP); the RTE gap is
attributable to the paper's MNLI→RTE intermediate fine-tuning trick, which we
do not replicate. The CLIP extension shows the same `LoRALinear` primitive
ports cleanly to a multi-tower vision-language model.

## 7. Conclusion

LoRA's design holds up exactly as advertised: ~0.3M trainable parameters
recover full fine-tuning quality, and `merge()` gives identical inference
cost to the unmodified base model. The paper's results are reproducible from
~200 lines of LoRA code, with the per-architecture quirks (GPT-2's `Conv1D`
transpose and fused QKV, multi-tower CLIP) absorbed by small layer-class
variants without changing the core algorithm.

## 8. References

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.,
  & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- Reference implementation: <https://github.com/microsoft/LoRA>.
- Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language
  Processing.* EMNLP System Demonstrations.
- Wang, A. et al. (2019). *GLUE: A Multi-Task Benchmark and Analysis Platform
  for Natural Language Understanding.* ICLR.
- Novikova, J., Dušek, O., & Rieser, V. (2017). *The E2E Dataset: New
  Challenges for End-to-End Generation.* SIGDIAL.
- Radford, A. et al. (2021). *Learning Transferable Visual Models from
  Natural Language Supervision* (CLIP). ICML.

## 9. Acknowledgements

This work was completed for **CS 4782: Introduction to Deep Learning**
(Cornell, Spring 2026). Thanks to the course staff for the project framing
and feedback. Compute was provided by Google Colab (T4 / L4 GPUs). All bugs
are ours; all credit for the LoRA method is the original authors'.

**Authors.** Austin Lu, Jason Chen.
**License.** [MIT](LICENSE).
