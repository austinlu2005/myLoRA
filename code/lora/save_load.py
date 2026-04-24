import torch

def save_lora_state_dict(model, path):
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state, path)

def load_lora_state_dict(model, path, strict=False):
    lora_state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    # Only fail if LoRA params are missing; base params are expected to be "missing"
    missing_lora = [k for k in missing if "lora_" in k]
    assert not missing_lora, f"Missing LoRA params: {missing_lora}"
    return model