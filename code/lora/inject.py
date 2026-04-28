import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from .layers import LoRAConv1D, LoRAConv1DQV, LoRALinear


def inject_lora(
    model,
    target_modules,
    rank=8,
    alpha=16,
    dropout=0.0,
    conv1d_qv=False,
    allowed_prefixes=None,
):
    """Walk the model and replace target modules with LoRA-wrapped versions.

    conv1d_qv: when True, c_attn Conv1D modules are wrapped with LoRAConv1DQV
    (independent adapters on Q and V slices, K untouched). Use this for GPT-2
    to match the LoRA paper's W_q, W_v target. Other Conv1D modules still use
    the plain LoRAConv1D.

    allowed_prefixes: optional list/tuple of module-name prefixes. When set,
    only matching modules whose full name starts with one of these prefixes
    will be LoRA-wrapped. This is useful for multimodal models where the same
    target names (for example q_proj / v_proj) appear in multiple towers.
    """
    allowed_prefixes = tuple(allowed_prefixes) if allowed_prefixes is not None else None
    replaced = []
    for name, module in model.named_modules():
        if not any(t in name for t in target_modules):
            continue
        if allowed_prefixes is not None and not name.startswith(allowed_prefixes):
            continue
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model

        if isinstance(module, nn.Linear):
            new_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        elif isinstance(module, Conv1D):
            if conv1d_qv and child_name == "c_attn":
                new_module = LoRAConv1DQV(module, rank=rank, alpha=alpha, dropout=dropout)
            else:
                new_module = LoRAConv1D(module, rank=rank, alpha=alpha, dropout=dropout)
        else:
            continue

        setattr(parent, child_name, new_module)
        replaced.append(name)

    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False

    return model, replaced
