import argparse
import math
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.optim import build_optimizer, build_scheduler
from training.trainer import Trainer
from utils.config import load_config
from utils.param_utils import print_trainable_parameters
from utils.seed import set_seed


def _build_roberta(cfg, tokenizer):
    from dataloaders.glue import load_glue
    from evaluation.glue_metrics import compute_glue_metrics
    from models.roberta_wrapper import build_roberta_lora

    model, replaced = build_roberta_lora(
        model_name=cfg["model"]["name"],
        num_labels=cfg["task"]["num_labels"],
        rank=cfg["lora"]["rank"],
        alpha=cfg["lora"]["alpha"],
        dropout=cfg["lora"]["dropout"],
    )
    data = load_glue(cfg["task"]["name"], tokenizer, max_length=cfg["task"]["max_length"])
    train_ds = data["train"]
    eval_split = "validation_matched" if cfg["task"]["name"] == "mnli" else "validation"
    eval_ds = data[eval_split]

    task_name = cfg["task"]["name"]

    def metrics_fn(preds, labels):
        return compute_glue_metrics(task_name, preds.numpy(), labels.numpy())

    return model, replaced, train_ds, eval_ds, metrics_fn, "classification"


def _build_gpt2(cfg, tokenizer):
    from models.gpt2_wrapper import build_gpt2_lora

    model, replaced = build_gpt2_lora(
        model_name=cfg["model"]["name"],
        rank=cfg["lora"]["rank"],
        alpha=cfg["lora"]["alpha"],
        dropout=cfg["lora"]["dropout"],
    )

    task_name = cfg["task"]["name"]
    if task_name == "e2e_nlg":
        from dataloaders.e2e_nlg import load_e2e_nlg

        data = load_e2e_nlg(tokenizer, max_length=cfg["task"]["max_length"])
        train_ds, eval_ds = data["train"], data["validation"]
    elif task_name == "wikitext":
        from dataloaders.wikitext import load_wikitext

        data = load_wikitext(
            tokenizer,
            block_size=cfg["task"]["max_length"],
            config=cfg["task"].get("config", "wikitext-2-raw-v1"),
        )
        train_ds, eval_ds = data["train"], data["validation"]
    else:
        raise NotImplementedError(f"gpt2 task {task_name!r} not supported")

    return model, replaced, train_ds, eval_ds, None, "causal_lm"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    model_type = cfg["model"]["type"]
    if model_type == "roberta":
        model, replaced, train_ds, eval_ds, metrics_fn, task_type = _build_roberta(cfg, tokenizer)
    elif model_type == "gpt2":
        model, replaced, train_ds, eval_ds, metrics_fn, task_type = _build_gpt2(cfg, tokenizer)
    else:
        raise NotImplementedError(f"model.type {model_type!r} not supported")

    print(f"Injected LoRA into {len(replaced)} modules")
    print_trainable_parameters(model)

    steps_per_epoch = math.ceil(len(train_ds) / cfg["training"]["batch_size"])
    num_training_steps = steps_per_epoch * cfg["training"]["num_epochs"]

    optimizer = build_optimizer(
        model,
        lr=float(cfg["training"]["lr"]),
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer,
        num_training_steps=num_training_steps,
        warmup_ratio=cfg["training"]["warmup_ratio"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=cfg["training"]["batch_size"],
        num_epochs=cfg["training"]["num_epochs"],
        device=device,
        compute_metrics=metrics_fn,
        log_steps=cfg["training"]["log_steps"],
        grad_clip=cfg["training"].get("grad_clip", 1.0),
        output_dir=cfg.get("output", {}).get("dir"),
        task_type=task_type,
    )
    trainer.train()
    print(f"best eval metric: {trainer.best_metric}")


if __name__ == "__main__":
    main()
