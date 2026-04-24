def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_trainable_parameters(model):
    trainable, total = count_parameters(model)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")
    return trainable, total


def freeze_base_model(model):
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False


def unfreeze_by_name(model, substrings):
    for n, p in model.named_parameters():
        if any(s in n for s in substrings):
            p.requires_grad = True
