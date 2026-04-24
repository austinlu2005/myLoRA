import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from .layers import LoRALinear, LoRAConv1D


def inject_lora(model, target_modules, rank=8, alpha=16, dropout=0.0):
    """Walk the model and replace target modules with LoRA-wrapped versions."""
    replaced = []
    for name, module in model.named_modules():
        if not any(t in name for t in target_modules):
            continue
        # Get parent module to do the replacement
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        
        if isinstance(module, nn.Linear):
            new_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        elif isinstance(module, Conv1D):
            new_module = LoRAConv1D(module, rank=rank, alpha=alpha, dropout=dropout)
        else:
            continue
        
        setattr(parent, child_name, new_module)
        replaced.append(name)
    
    # Freeze everything else
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    
    return model, replaced