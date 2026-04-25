import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from .layers import LoRAConv1D, LoRAConv1DQV, LoRALinear


def inject_lora(model, target_modules, rank=8, alpha=16, dropout=0.0, conv1d_qv=False):
    """Walk the model and replace target modules with LoRA-wrapped versions.

    conv1d_qv: when True, c_attn Conv1D modules are wrapped with LoRAConv1DQV
    (independent adapters on Q and V slices, K untouched). Use this for GPT-2
    to match the LoRA paper's W_q, W_v target. Other Conv1D modules still use
    the plain LoRAConv1D.
    """
    replaced = []
    for name, module in model.named_modules():
        if not any(t in name for t in target_modules):
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
