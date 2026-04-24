from datasets import load_dataset


GLUE_TASK_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def load_glue(task, tokenizer, max_length=128, cache_dir=None):
    if task not in GLUE_TASK_KEYS:
        raise KeyError(f"Unknown GLUE task '{task}'. Known: {sorted(GLUE_TASK_KEYS)}")

    raw = load_dataset("glue", task, cache_dir=cache_dir)
    key_a, key_b = GLUE_TASK_KEYS[task]

    def tokenize(batch):
        if key_b is None:
            return tokenizer(
                batch[key_a],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        return tokenizer(
            batch[key_a],
            batch[key_b],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = raw.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    keep_cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized["train"].column_names:
        keep_cols.append("token_type_ids")

    tokenized.set_format("torch", columns=keep_cols)
    return tokenized
