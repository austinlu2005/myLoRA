from .layers import LoRAConv1D, LoRAConv1DQV, LoRALinear


_LORA_TYPES = (LoRALinear, LoRAConv1D, LoRAConv1DQV)


def merge_lora(model):
    for m in model.modules():
        if isinstance(m, _LORA_TYPES) and hasattr(m, "merge"):
            m.merge()
    return model


def unmerge_lora(model):
    for m in model.modules():
        if isinstance(m, _LORA_TYPES) and hasattr(m, "unmerge"):
            m.unmerge()
    return model
