# Target patterns per architecture. These are substring matches against module names.
TARGET_MODULES = {
    "roberta": ["query", "value"],  # paper default for BERT/RoBERTa
    "gpt2":    ["c_attn"],           # GPT-2 fuses q,k,v into one Conv1D
    "llama":   ["q_proj", "v_proj"],
    "llava":   ["q_proj", "v_proj"], # typically only language tower
    "clip_vision": ["q_proj", "v_proj"],
}


def get_target_modules(model_type: str, include_mlp: bool = False):
    targets = TARGET_MODULES[model_type].copy()
    if include_mlp:
        mlp_targets = {
            "roberta": ["intermediate.dense", "output.dense"],
            "gpt2": ["c_fc", "c_proj"],
            "llama": ["gate_proj", "up_proj", "down_proj"],
        }
        targets.extend(mlp_targets.get(model_type, []))
    return targets