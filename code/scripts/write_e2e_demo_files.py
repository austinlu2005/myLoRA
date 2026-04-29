"""Generate a tiny E2E NLG sample and write predictions.txt / references.txt.

This is a 3-example version of the full E2E evaluation flow. It uses the
same text-file format as the official scorer:
  - predictions.txt: one prediction per MR
  - references.txt: one or more references per MR, blank line between groups

By default the script tries to use the saved GPT-2 Medium LoRA checkpoint in
results/GPT2-M/e2e_nlg/. If none is found, it falls back to plain GPT-2 Medium.

Example:
    python code/scripts/write_e2e_demo_files.py
    python code/scripts/write_e2e_demo_files.py --num-examples 3 --output-dir ./demo_e2e
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
from evaluation.generation_metrics import aggregate_test_set, generate_e2e_predictions, write_e2e_files
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


def build_model_and_tokenizer(model_name, lora_path, rank, alpha, dropout):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora_path is not None:
        model, _ = build_gpt2_lora(
            model_name=model_name,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        load_lora_state_dict(model, lora_path)
        merge_lora(model)
        label = f"LoRA checkpoint ({lora_path})"
    else:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        label = f"base model ({model_name})"

    return tokenizer, model, label


def main():
    parser = argparse.ArgumentParser(
        description="Write predictions.txt and references.txt for a tiny E2E GPT-2 demo."
    )
    parser.add_argument("--model-name", default="gpt2-medium")
    parser.add_argument("--lora-path", default=None, help="Optional path to lora_best.pt")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./demo_e2e")
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
    indices = sorted(indices[:count])

    selected_mrs = [mrs[i] for i in indices]
    selected_refs = [refs[i] for i in indices]

    lora_path = resolve_lora_path(args.lora_path)
    tokenizer, model, label = build_model_and_tokenizer(
        model_name=args.model_name,
        lora_path=lora_path,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"device: {device}")
    print(f"dataset split: {args.split}")
    print(f"model: {label}")
    print(f"writing {count} examples")

    predictions = generate_e2e_predictions(
        model,
        tokenizer,
        selected_mrs,
        prompt_sep=PROMPT_SEP,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    out_dir = Path(args.output_dir)
    pred_path, refs_path = write_e2e_files(predictions, selected_refs, out_dir)

    print(f"\nwrote: {pred_path}")
    print(f"wrote: {refs_path}")
    print("\nexamples:")
    for i, (mr, pred, ref_group) in enumerate(zip(selected_mrs, predictions, selected_refs), start=1):
        print(f"\n=== Example {i} ===")
        print(f"MR: {mr}")
        print(f"Prediction: {pred}")
        print(f"Reference 1: {ref_group[0]}")


if __name__ == "__main__":
    main()
