# Datasets

This project does not ship any raw datasets. Everything is fetched on demand
from the Hugging Face Hub the first time the relevant training script runs and
cached under `~/.cache/huggingface/datasets/` (override with `HF_DATASETS_CACHE`).

## GLUE — RoBERTa-base experiments

Loaded via `datasets.load_dataset("glue", TASK)`.

| Task  | Purpose                     | Paper metric        |
|-------|-----------------------------|---------------------|
| mnli  | natural language inference  | accuracy (matched)  |
| sst2  | sentiment                   | accuracy            |
| mrpc  | paraphrase                  | accuracy            |
| cola  | linguistic acceptability    | Matthews corr.      |
| qnli  | QA inference                | accuracy            |
| qqp   | paraphrase                  | accuracy            |
| rte   | textual entailment          | accuracy            |
| stsb  | semantic similarity         | Pearson corr.       |

Pre-cache all eight in one shot:

```python
from datasets import load_dataset
for task in ["mrpc", "sst2", "cola", "rte", "stsb", "qnli", "mnli", "qqp"]:
    load_dataset("glue", task)
```

## E2E NLG Challenge — GPT-2 Medium experiments

`datasets.load_dataset("e2e_nlg", revision="refs/convert/parquet")` — the
parquet mirror keeps it compatible with `datasets >= 4.0`. Each example pairs
a meaning representation with one or more human references; we fine-tune as
causal LM with the prompt tokens masked. Multi-reference test scoring uses the
official `e2e-metrics` scorer (BLEU / NIST / METEOR / ROUGE-L / CIDEr) —
install separately from <https://github.com/tuetschek/e2e-metrics>.

## WikiText — GPT-2 ablations

`datasets.load_dataset("wikitext", "wikitext-2-raw-v1")`. Language-modeling
perplexity sanity check in the GPT-2 notebook.

## Pokémon BLIP captions — CLIP vision-language extension

`datasets.load_dataset("lambdalabs/pokemon-blip-captions")` — ~830
image-caption pairs. Used to fine-tune CLIP-ViT-Base/32 with LoRA adapters on
attention Q/V projections in both towers, evaluated as image↔text retrieval
Recall@K.

## No local storage of datasets

Raw data is intentionally not committed — the GLUE caches alone exceed 1 GB
and the Hub is the authoritative source.
