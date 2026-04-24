# Datasets

This project does not ship any raw datasets. All data is fetched on-demand from
the Hugging Face Hub the first time training runs.

## GLUE (RoBERTa experiments)

Loaded via `datasets.load_dataset("glue", TASK)` where `TASK` is one of:

| Task  | Purpose | Paper metric |
|-------|---------|--------------|
| mnli  | natural language inference | accuracy (matched) |
| sst2  | sentiment                  | accuracy |
| mrpc  | paraphrase                 | accuracy |
| cola  | linguistic acceptability   | Matthews corr. |
| qnli  | QA inference               | accuracy |
| qqp   | paraphrase                 | accuracy |
| rte   | textual entailment         | accuracy |
| stsb  | semantic similarity        | Pearson corr. |

First download runs from the machine that trains; `datasets` caches under
`~/.cache/huggingface/datasets/` (or `HF_DATASETS_CACHE`).

## How to trigger the download manually

```python
from datasets import load_dataset
for task in ["mrpc", "sst2", "cola", "rte", "stsb", "qnli", "mnli", "qqp"]:
    load_dataset("glue", task)
```

## No local storage of datasets

Raw data is intentionally not committed to this repo — it would add ~1 GB for
GLUE alone and the HF Hub is the authoritative source.
