from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def build_optimizer(model, lr, weight_decay=0.0, betas=(0.9, 0.999)):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return AdamW(trainable, lr=lr, betas=betas, weight_decay=weight_decay)


def build_scheduler(optimizer, num_training_steps, warmup_ratio=None, warmup_steps=None):
    """Linear schedule with warmup. Prefers `warmup_steps` (absolute, paper-exact for E2E)
    over `warmup_ratio` (fraction of total steps) when both are provided. Defaults to
    `warmup_ratio=0.06` if neither is set.
    """
    if warmup_steps is None:
        if warmup_ratio is None:
            warmup_ratio = 0.06
        warmup_steps = int(num_training_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
