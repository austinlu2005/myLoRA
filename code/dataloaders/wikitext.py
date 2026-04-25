from datasets import load_dataset


def load_wikitext(tokenizer, block_size=512, config="wikitext-2-raw-v1", cache_dir=None):
    """Load WikiText for causal-LM fine-tuning of GPT-2.

    Concatenates all text in each split, then chunks into fixed `block_size`
    blocks. labels = input_ids; GPT-2's lm head handles the shift.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("wikitext", config, cache_dir=cache_dir)

    def tokenize(batch):
        return tokenizer(batch["text"], add_special_tokens=False)

    tokenized = raw.map(
        tokenize,
        batched=True,
        remove_columns=raw["train"].column_names,
    )

    def group(batch):
        concat = sum(batch["input_ids"], [])
        total = (len(concat) // block_size) * block_size
        chunks = [concat[i : i + block_size] for i in range(0, total, block_size)]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * block_size for _ in chunks],
            "labels": [list(c) for c in chunks],
        }

    grouped = tokenized.map(
        group,
        batched=True,
        remove_columns=tokenized["train"].column_names,
    )
    grouped.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return grouped
