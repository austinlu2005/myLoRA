"""One-shot probe of nanoVLM + the two VQA datasets.

Goal: gather everything we need to design a LoRA wrapper for nanoVLM
without committing to an API guess. Run this once and paste the output back.

Usage:
    python scripts/probe_nanovlm.py
    # or paste the body of main() into a Colab cell after `pip install -q transformers datasets pillow`.
"""
from __future__ import annotations

import inspect
import sys
import traceback
from collections import defaultdict


CHECKPOINT = "HuggingFaceTB/SmolVLM-256M-Instruct"
AOKVQA_CANDIDATES = ["HuggingFaceM4/A-OKVQA", "allenai/aokvqa"]
VIZWIZ_CANDIDATES = ["Multimodal-Fatima/VizWiz_VQA", "lmms-lab/VizWiz-VQA", "HuggingFaceM4/VizWiz"]


def banner(s: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {s}")
    print("=" * 78)


# ------------------------------------------------------------------ load model
def try_load_model():
    """Try multiple loading strategies; return (model, source_label)."""
    import torch
    from transformers import AutoConfig

    # 0. Inspect the config first — tells us what model_type / architecture is registered.
    try:
        cfg = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True)
        print(f"AutoConfig OK: model_type={getattr(cfg, 'model_type', '?')} | "
              f"architectures={getattr(cfg, 'architectures', '?')}")
    except Exception as e:
        print(f"AutoConfig FAILED: {type(e).__name__}: {e}")
        cfg = None

    # 1. Try Auto* classes most likely to wire it up correctly.
    attempts = []
    try:
        from transformers import AutoModelForVision2Seq
        attempts.append(("AutoModelForVision2Seq", AutoModelForVision2Seq))
    except ImportError:
        pass
    try:
        from transformers import AutoModelForImageTextToText
        attempts.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
    except ImportError:
        pass
    from transformers import AutoModelForCausalLM, AutoModel
    attempts.append(("AutoModelForCausalLM", AutoModelForCausalLM))
    attempts.append(("AutoModel", AutoModel))

    for label, cls in attempts:
        try:
            model = cls.from_pretrained(CHECKPOINT, trust_remote_code=True)
            print(f"LOAD OK via {label}")
            return model, label
        except Exception as e:
            print(f"LOAD FAIL via {label}: {type(e).__name__}: {str(e)[:200]}")

    # 2. Fallback: try the project's own VisionLanguageModel class if the repo ships one.
    try:
        sys.path.insert(0, "/tmp")
        import subprocess
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/huggingface/nanoVLM.git", "/tmp/nanoVLM"],
            check=False, capture_output=True,
        )
        sys.path.insert(0, "/tmp/nanoVLM")
        from models.vision_language_model import VisionLanguageModel  # type: ignore
        model = VisionLanguageModel.from_pretrained(CHECKPOINT)
        print("LOAD OK via huggingface/nanoVLM:VisionLanguageModel")
        return model, "VisionLanguageModel"
    except Exception as e:
        print(f"LOAD FAIL via VisionLanguageModel fallback: {type(e).__name__}: {str(e)[:200]}")

    return None, None


def summarize_modules(model) -> None:
    import torch.nn as nn

    banner("Top-level children")
    for name, child in model.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        print(f"  {name:30s} {type(child).__name__:30s} {n_params/1e6:7.2f}M params")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total/1e6:.2f}M | trainable now: {trainable/1e6:.2f}M")

    # Look for attention projection patterns.
    banner("Attention projection patterns (paths grouped by parent)")
    patterns = ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj",
                "c_attn", "c_proj", "wq", "wk", "wv", "wo",
                "query", "key", "value"]
    by_parent = defaultdict(list)
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in patterns and isinstance(mod, (nn.Linear,)):
            parent = ".".join(name.split(".")[:-1])
            by_parent[parent].append(f"{leaf}({tuple(mod.weight.shape)})")
    if not by_parent:
        print("  (no leaves matched standard patterns — print all nn.Linear instead:)")
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                print(f"  {name:60s} {tuple(mod.weight.shape)}")
    else:
        # Show 2 examples per distinct parent path stem so we don't dump all 12 layers.
        seen_stems = defaultdict(int)
        for parent, leaves in by_parent.items():
            stem = ".".join(p for p in parent.split(".") if not p.isdigit())
            seen_stems[stem] += 1
            if seen_stems[stem] > 2:
                continue
            print(f"  {parent}")
            for leaf in leaves:
                print(f"    └─ {leaf}")
        print(f"\n  (showing ≤2 examples per stem; total {sum(len(v) for v in by_parent.values())} matching projections "
              f"across {len(by_parent)} parents)")

    # Top-level forward signature.
    banner("Forward signature (top-level model)")
    try:
        sig = inspect.signature(model.forward)
        print(f"  {sig}")
    except Exception as e:
        print(f"  could not inspect: {e}")


def try_processor_and_tokenizer():
    banner("Processor / tokenizer")
    out = {}
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
        print(f"AutoProcessor OK: {type(proc).__name__}")
        out["processor"] = proc
    except Exception as e:
        print(f"AutoProcessor FAIL: {type(e).__name__}: {str(e)[:200]}")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
        print(f"AutoTokenizer OK: {type(tok).__name__} | vocab={tok.vocab_size} | pad={tok.pad_token!r}")
        out["tokenizer"] = tok
    except Exception as e:
        print(f"AutoTokenizer FAIL: {type(e).__name__}: {str(e)[:200]}")
    try:
        from transformers import AutoImageProcessor
        ip = AutoImageProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
        print(f"AutoImageProcessor OK: {type(ip).__name__} | size={getattr(ip, 'size', '?')}")
        out["image_processor"] = ip
    except Exception as e:
        print(f"AutoImageProcessor FAIL: {type(e).__name__}: {str(e)[:200]}")
    return out


def smoke_forward(model, helpers):
    banner("Smoke forward pass (dummy image + 'Question: hi\\nAnswer:')")
    import torch
    from PIL import Image

    img = Image.new("RGB", (256, 256), color=(127, 127, 127))
    text = "Question: hi\nAnswer:"

    candidates = []  # list of (label, kwargs_dict_factory)

    proc = helpers.get("processor")
    if proc is not None:
        try:
            batch = proc(images=img, text=text, return_tensors="pt")
            candidates.append(("processor(images=, text=)", batch))
        except Exception as e:
            print(f"  processor(images=, text=) FAIL: {type(e).__name__}: {str(e)[:200]}")

    ip = helpers.get("image_processor")
    tok = helpers.get("tokenizer")
    if ip is not None and tok is not None:
        try:
            pix = ip(images=img, return_tensors="pt")
            ids = tok(text, return_tensors="pt")
            batch = {**pix, **ids}
            candidates.append(("image_processor + tokenizer", batch))
        except Exception as e:
            print(f"  image_processor + tokenizer FAIL: {type(e).__name__}: {str(e)[:200]}")

    if not candidates:
        print("  (no input pipeline could be built — model needs custom preprocessing)")
        return

    for label, batch in candidates:
        print(f"\n  trying: {label}")
        print(f"  batch keys: {list(batch.keys())}")
        for k, v in batch.items():
            shape = tuple(getattr(v, "shape", ()))
            print(f"    {k}: shape={shape} dtype={getattr(v, 'dtype', '?')}")
        try:
            with torch.no_grad():
                out = model(**batch)
            has_logits = hasattr(out, "logits")
            has_loss = hasattr(out, "loss") and getattr(out, "loss", None) is not None
            logits_shape = tuple(out.logits.shape) if has_logits else None
            print(f"  forward OK | type={type(out).__name__} | logits_shape={logits_shape} | "
                  f"loss_present_without_labels={has_loss}")
        except Exception as e:
            print(f"  forward FAIL: {type(e).__name__}: {str(e)[:250]}")
            continue

        # Try with labels to see if the model computes LM loss internally.
        if "input_ids" in batch:
            labels_batch = {**batch, "labels": batch["input_ids"].clone()}
            try:
                with torch.no_grad():
                    out2 = model(**labels_batch)
                print(f"  forward(labels=) OK | loss={getattr(out2, 'loss', None)}")
            except Exception as e:
                print(f"  forward(labels=) FAIL: {type(e).__name__}: {str(e)[:250]}")
        break  # one successful pipeline is enough


# ----------------------------------------------------------------- datasets
def probe_dataset(candidates):
    from datasets import load_dataset

    for name in candidates:
        print(f"\n  trying load_dataset({name!r})")
        try:
            ds = load_dataset(name, split="train[:5]")
            print(f"  OK: split=train[:5] | columns={ds.column_names}")
            ex = ds[0]
            for k, v in ex.items():
                if hasattr(v, "size") and hasattr(v, "mode"):  # PIL image
                    print(f"    {k}: <PIL.Image size={v.size} mode={v.mode}>")
                elif isinstance(v, (list, tuple)) and len(v) > 8:
                    print(f"    {k}: <{type(v).__name__} len={len(v)}> head={v[:3]}")
                elif isinstance(v, str) and len(v) > 200:
                    print(f"    {k}: <str len={len(v)}> head={v[:200]!r}")
                else:
                    print(f"    {k}: {v!r}")
            return name
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {str(e)[:240]}")
    return None


def main():
    banner(f"Loading {CHECKPOINT}")
    model, source = try_load_model()
    if model is None:
        print("\nABORTING: could not load model. Paste the output above and we'll figure it out.")
        return

    summarize_modules(model)
    helpers = try_processor_and_tokenizer()
    try:
        smoke_forward(model, helpers)
    except Exception:
        traceback.print_exc()

    banner("A-OKVQA dataset probe")
    aokvqa_used = probe_dataset(AOKVQA_CANDIDATES)
    banner("VizWiz dataset probe")
    vizwiz_used = probe_dataset(VIZWIZ_CANDIDATES)

    banner("SUMMARY (paste this top section back to me at minimum)")
    print(f"  load_source       : {source}")
    print(f"  model class       : {type(model).__name__}")
    print(f"  aokvqa dataset    : {aokvqa_used}")
    print(f"  vizwiz dataset    : {vizwiz_used}")


if __name__ == "__main__":
    main()
