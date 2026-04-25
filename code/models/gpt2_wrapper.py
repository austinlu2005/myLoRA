from transformers import GPT2LMHeadModel

from lora.inject import inject_lora
from lora.targets import get_target_modules


def build_gpt2_lora(model_name, rank, alpha, dropout=0.1):
    """Load GPT-2 (or GPT-2 medium/large) with LoRA adapters on c_attn Q/V slices."""
    model = GPT2LMHeadModel.from_pretrained(model_name)
    targets = get_target_modules("gpt2")
    model, replaced = inject_lora(
        model,
        targets,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        conv1d_qv=True,
    )
    # GPT-2 ties lm_head to wte; both stay frozen with the rest of the base.
    return model, replaced
