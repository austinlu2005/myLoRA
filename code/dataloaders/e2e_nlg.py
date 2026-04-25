from datasets import load_dataset


PROMPT_SEP = " ||| "


def load_e2e_nlg(tokenizer, max_length=512, cache_dir=None, dataset_name="e2e_nlg"):
    """Load E2E NLG for causal-LM fine-tuning of GPT-2.

    Each example is formatted as `<MR> ||| <reference><eos>`. Prompt tokens
    (the MR plus separator) are masked to -100 so loss is only computed over
    the reference completion. GPT-2's lm head shifts labels internally.

    Loads from the auto-generated parquet mirror at `refs/convert/parquet`
    so it works on `datasets >= 4.0`, which dropped script-based loaders.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        raw = load_dataset(dataset_name, revision="refs/convert/parquet", cache_dir=cache_dir)
    except Exception:
        raw = load_dataset(dataset_name, cache_dir=cache_dir)
    pad_id = tokenizer.pad_token_id
    eos = tokenizer.eos_token

    def encode(example):
        mr = example["meaning_representation"]
        ref = example["human_reference"]
        prompt = f"{mr}{PROMPT_SEP}"
        full = prompt + ref + eos

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            full,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + list(full_ids[prompt_len:])
        attention_mask = [1] * len(full_ids)

        pad_n = max_length - len(full_ids)
        if pad_n > 0:
            full_ids = full_ids + [pad_id] * pad_n
            attention_mask = attention_mask + [0] * pad_n
            labels = labels + [-100] * pad_n

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized = raw.map(encode, remove_columns=raw["train"].column_names)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized
