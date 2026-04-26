"""Generation metrics for the E2E NLG Challenge.

Two scoring backends:
- "pip"      : sacrebleu / nltk / rouge_score / pycocoevalcap (default; pure pip)
- "official" : subprocesses tuetschek/e2e-metrics measure_scores.py (paper-exact)

The pip stack reproduces paper numbers to within ~0.1-0.5 BLEU due to
tokenization differences. Use the official path when you need to match the
LoRA paper's Table 3 numbers exactly.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch
from tqdm.auto import tqdm


def aggregate_test_set(test_dataset):
    """Group references by MR (preserves first-occurrence order).

    Returns: (unique_mrs: list[str], refs_per_mr: list[list[str]])
    """
    refs_by_mr = OrderedDict()
    for ex in test_dataset:
        mr = ex["meaning_representation"]
        ref = ex["human_reference"]
        refs_by_mr.setdefault(mr, []).append(ref)
    return list(refs_by_mr.keys()), list(refs_by_mr.values())


def generate_e2e_predictions(
    model,
    tokenizer,
    mrs,
    prompt_sep=" ||| ",
    beam_size=10,
    length_penalty=0.9,
    no_repeat_ngram_size=4,
    max_new_tokens=80,
    device="cuda",
):
    """Beam-search one prediction per MR. Returns list[str] aligned with `mrs`."""
    model.eval()
    predictions = []
    for mr in tqdm(mrs, desc="generate"):
        prompt = f"{mr}{prompt_sep}"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                num_beams=beam_size,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        # E2E refs are single sentences; cut at first newline as a safety stop
        completion = completion.split("\n")[0].strip()
        predictions.append(completion)
    return predictions


def write_e2e_files(predictions, refs_groups, out_dir):
    """Write predictions.txt + references.txt in tuetschek/e2e-metrics format."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.txt"
    refs_path = out_dir / "references.txt"
    with pred_path.open("w") as f:
        for p in predictions:
            f.write(p.strip() + "\n")
    with refs_path.open("w") as f:
        for refs in refs_groups:
            for r in refs:
                f.write(r.strip() + "\n")
            f.write("\n")  # blank line separates MR groups
    return pred_path, refs_path


# --------------------------------------------------------------- official scorer
def score_e2e_official(pred_path, refs_path, e2e_metrics_path):
    """Run tuetschek/e2e-metrics measure_scores.py and parse its output."""
    script = Path(e2e_metrics_path) / "measure_scores.py"
    if not script.exists():
        raise FileNotFoundError(f"measure_scores.py not found at {script}")

    cmd = [sys.executable, str(script), str(refs_path), str(pred_path)]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"measure_scores.py exited with code {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n"
            f"To debug, run the command above directly in a shell."
        )

    metrics = {}
    for line in result.stdout.splitlines():
        m = re.match(r"^([A-Za-z_]+):\s+([\d.]+)\s*$", line.strip())
        if m:
            key = m.group(1).upper().replace("CIDER", "CIDEr").replace("ROUGE_L", "ROUGE_L")
            val = float(m.group(2))
            if key in {"BLEU", "NIST", "METEOR", "ROUGE_L", "CIDEr"}:
                metrics[key] = val
    if len(metrics) < 5:
        raise RuntimeError(
            f"Failed to parse all 5 metrics from measure_scores.py output.\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return metrics


# ------------------------------------------------------------------- pip scorer
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def score_e2e_pip(predictions, refs_groups):
    """Pip-only fallback. Uses nltk/sacrebleu/rouge_score/pycocoevalcap.

    Numbers may differ from paper by ~0.1-0.5 BLEU due to tokenization.
    """
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    from nltk.translate import bleu_score, nist_score, meteor_score

    pred_norm = [_normalize(p) for p in predictions]
    refs_norm = [[_normalize(r) for r in g] for g in refs_groups]
    pred_toks = [nltk.word_tokenize(p) for p in pred_norm]
    refs_toks = [[nltk.word_tokenize(r) for r in g] for g in refs_norm]

    bleu = bleu_score.corpus_bleu(refs_toks, pred_toks)
    try:
        nist = nist_score.corpus_nist(refs_toks, pred_toks)
    except Exception:
        nist = float("nan")

    meteor = sum(
        meteor_score.meteor_score(refs_toks[i], pred_toks[i])
        for i in range(len(predictions))
    ) / max(len(predictions), 1)

    from rouge_score import rouge_scorer
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = sum(
        max(rouge.score(r, predictions[i])["rougeL"].fmeasure for r in refs_groups[i])
        for i in range(len(predictions))
    ) / max(len(predictions), 1)

    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {i: refs_groups[i] for i in range(len(predictions))}
        res = {i: [predictions[i]] for i in range(len(predictions))}
        cider, _ = Cider().compute_score(gts, res)
    except Exception as e:
        print(f"  CIDEr unavailable ({type(e).__name__}: {e}); reporting NaN")
        cider = float("nan")

    return {
        "BLEU": bleu,
        "NIST": nist,
        "METEOR": meteor,
        "ROUGE_L": rouge_l,
        "CIDEr": cider,
    }


# ---------------------------------------------------------------- entry point
def score_e2e(predictions, refs_groups, e2e_metrics_path=None, work_dir=None):
    """Compute the 5 E2E NLG metrics. Returns dict.

    If e2e_metrics_path is set, use the official scorer (paper-aligned).
    Otherwise fall back to a pip-installable stack.
    """
    if e2e_metrics_path:
        if work_dir is None:
            work_dir = tempfile.mkdtemp()
        pred_path, refs_path = write_e2e_files(predictions, refs_groups, work_dir)
        return score_e2e_official(pred_path, refs_path, e2e_metrics_path)
    return score_e2e_pip(predictions, refs_groups)


# Paper Table 3 row for GPT-2 Medium + LoRA (rank 4) on E2E NLG test set.
PAPER_E2E_LORA = {
    "BLEU": 0.704,
    "NIST": 8.85,
    "METEOR": 0.460,
    "ROUGE_L": 0.718,
    "CIDEr": 2.53,
}
