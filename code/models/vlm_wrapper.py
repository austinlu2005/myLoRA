from transformers import CLIPModel

from lora.inject import inject_lora
from lora.targets import get_target_modules


_CLIP_ALLOWED_PREFIXES = {
    "vision": ["vision_model.encoder.layers"],
    "text": ["text_model.encoder.layers"],
    "both": ["vision_model.encoder.layers", "text_model.encoder.layers"],
}


def build_clip_lora(model_name, rank, alpha, dropout=0.1, tower="both"):
    """Load CLIP with LoRA adapters on attention Q/V projections.

    This extends the repo's LoRA implementation to a compact VLM that fits a
    T4/L4 workflow. By default both the text and vision towers receive LoRA
    adapters, while the pretrained projection heads and temperature stay frozen.
    """
    if tower not in _CLIP_ALLOWED_PREFIXES:
        raise ValueError(f"tower must be one of {sorted(_CLIP_ALLOWED_PREFIXES)}")

    model = CLIPModel.from_pretrained(model_name)
    targets = get_target_modules("clip")
    model, replaced = inject_lora(
        model,
        targets,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        allowed_prefixes=_CLIP_ALLOWED_PREFIXES[tower],
    )
    return model, replaced
