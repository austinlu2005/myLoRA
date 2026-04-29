"""Print a small E2E NLG demo for GPT-2 / GPT-2+LoRA.

By default this script tries to use the repo's saved GPT-2 Medium LoRA
checkpoint if it exists. Otherwise it falls back to plain GPT-2 Medium.

Example:
    python scripts/demo_e2e_gpt2.py
    python scripts/demo_e2e_gpt2.py --num-examples 5 --split test
    python scripts/demo_e2e_gpt2.py --lora-path results/GPT2-M/e2e_nlg/lora_best.pt
"""
import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

from dataloaders.e2e_nlg import PROMPT_SEP
from evaluation.generation_metrics import aggregate_test_set
from lora.merge import merge_lora
from lora.save_load import load_lora_state_dict
from models.gpt2_wrapper import build_gpt2_lora


DEFAULT_LORA_CANDIDATES = [
    REPO_ROOT / "results" / "GPT2-M" / "e2e_nlg" / "lora_best.pt",
    REPO_ROOT / "results" / "GPT2-M" / "e2e_nlg" / "lora_best_new.pt",
]


def resolve_lora_path(explicit_path: str | None):
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {path}")
        return path

    for candidate in DEFAULT_LORA_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_e2e_split(split: str):
    try:
        return load_dataset("e2e_nlg", split=split, revision="refs/convert/parquet")
    except Exception:
        return load_dataset("e2e_nlg", split=split)


def build_demo_model(model_name, lora_path, rank, alpha, dropout):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora_path is not None:
        model, replaced = build_gpt2_lora(
            model_name=model_name,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        load_lora_state_dict(model, lora_path)
        merge_lora(model)
        label = f"LoRA checkpoint ({lora_path})"
        return tokenizer, model, label, replaced

    model = GPT2LMHeadModel.from_pretrained(model_name)
    label = f"base model ({model_name})"
    return tokenizer, model, label, []


def generate_one(model, tokenizer, mr, beam_size, length_penalty, no_repeat_ngram_size, max_new_tokens, device):
    prompt = f"{mr}{PROMPT_SEP}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    return completion.split("\n")[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Print a GPT-2 E2E NLG demo.")
    parser.add_argument("--model-name", default="gpt2-medium")
    parser.add_argument("--lora-path", default=None, help="Optional path to lora_best.pt")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--length-penalty", type=float, default=0.9)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    dataset = load_e2e_split(args.split)
    mrs, refs = aggregate_test_set(dataset)
    count = min(args.num_examples, len(mrs))

    rng = random.Random(args.seed)
    indices = list(range(len(mrs)))
    rng.shuffle(indices)
    indices = indices[:count]

    lora_path = resolve_lora_path(args.lora_path)
    tokenizer, model, model_label, replaced = build_demo_model(
        model_name=args.model_name,
        lora_path=lora_path,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    print(f"device: {device}")
    print(f"dataset split: {args.split} | unique MRs: {len(mrs)}")
    print(f"model: {model_label}")
    if replaced:
        print(f"LoRA-injected modules: {len(replaced)}")
    print()

    for demo_idx, mr_idx in enumerate(indices, start=1):
        mr = mrs[mr_idx]
        references = refs[mr_idx]
        prediction = generate_one(
            model,
            tokenizer,
            mr,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

        print(f"=== Example {demo_idx} / {count} ===")
        print(f"MR: {mr}")
        print(f"Prompt: {mr}{PROMPT_SEP}")
        print("References:")
        for ref_idx, ref in enumerate(references, start=1):
            print(f"  {ref_idx}. {ref}")
        print("Model output:")
        print(f"  {prediction}")
        print()


if __name__ == "__main__":
    main()
