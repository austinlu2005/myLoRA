"""Generate E2E NLG predictions from a LoRA checkpoint and score against the
multi-reference test set. Reproduces the GPT-2 Medium + LoRA row of Table 3.

Usage:
    python scripts/evaluate_e2e.py --lora-path /path/to/lora_best.pt
    python scripts/evaluate_e2e.py --lora-path lora_best.pt --e2e-metrics /path/to/e2e-metrics
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from dataloaders.e2e_nlg import PROMPT_SEP
from evaluation.generation_metrics import (
    PAPER_E2E_LORA,
    aggregate_test_set,
    generate_e2e_predictions,
    score_e2e,
    write_e2e_files,
)
from lora.merge import merge_lora
from lora.save_load import load_lora_state_dict
from models.gpt2_wrapper import build_gpt2_lora


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora-path", required=True, help="Path to lora_best.pt")
    p.add_argument("--model-name", default="gpt2-medium")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--output-dir", default="./e2e_eval")
    p.add_argument("--e2e-metrics", default=None,
                   help="Path to cloned tuetschek/e2e-metrics repo. If unset, uses pip stack.")
    p.add_argument("--max-eval", type=int, default=0,
                   help="Limit number of unique MRs evaluated (0 = all)")
    p.add_argument("--beam-size", type=int, default=10)
    p.add_argument("--length-penalty", type=float, default=0.9)
    p.add_argument("--no-repeat-ngram-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=80)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. Tokenizer + model + LoRA adapter
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, replaced = build_gpt2_lora(
        model_name=args.model_name,
        rank=args.rank, alpha=args.alpha, dropout=args.dropout,
    )
    print(f"LoRA injected into {len(replaced)} c_attn modules")
    load_lora_state_dict(model, args.lora_path)
    print(f"Loaded adapter: {args.lora_path}")

    # 2. Merge for fast inference, move to device
    merge_lora(model)
    model = model.to(device)

    # 3. Load test split
    try:
        test = load_dataset("e2e_nlg", split="test", revision="refs/convert/parquet")
    except Exception:
        test = load_dataset("e2e_nlg", split="test")
    print(f"Test rows: {len(test)}")

    mrs, refs = aggregate_test_set(test)
    n_refs = [len(r) for r in refs]
    print(f"Unique MRs: {len(mrs)} | refs/MR min={min(n_refs)} max={max(n_refs)} avg={sum(n_refs)/len(n_refs):.1f}")

    if args.max_eval and args.max_eval < len(mrs):
        mrs = mrs[:args.max_eval]
        refs = refs[:args.max_eval]
        print(f"Truncated eval to first {len(mrs)} MRs")

    # 4. Generate
    preds = generate_e2e_predictions(
        model, tokenizer, mrs,
        prompt_sep=PROMPT_SEP,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    # 5. Persist outputs for inspection / re-scoring
    write_e2e_files(preds, refs, out_dir)
    print(f"Wrote {out_dir}/predictions.txt and references.txt")

    # 6. Score
    backend = "official" if args.e2e_metrics else "pip"
    print(f"Scoring backend: {backend}")
    metrics = score_e2e(preds, refs, e2e_metrics_path=args.e2e_metrics, work_dir=out_dir)

    # 7. Print + save
    print("\n=== Metrics ===")
    for k in ("BLEU", "NIST", "METEOR", "ROUGE_L", "CIDEr"):
        ours = metrics.get(k)
        paper = PAPER_E2E_LORA[k]
        scale = 100 if k in ("BLEU", "METEOR", "ROUGE_L") else 1
        if ours is None:
            print(f"  {k:8s}  ours: --       paper: {paper*scale:.2f}")
        else:
            print(f"  {k:8s}  ours: {ours*scale:7.2f}  paper: {paper*scale:7.2f}  Δ: {(ours-paper)*scale:+.2f}")

    results = {
        "metrics": metrics,
        "paper": PAPER_E2E_LORA,
        "config": vars(args),
        "n_unique_mrs": len(mrs),
        "scoring_backend": backend,
    }
    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_dir}/results.json")


if __name__ == "__main__":
    main()
