import argparse
import math
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.glue import load_glue
from evaluation.glue_metrics import compute_glue_metrics
from models.roberta_wrapper import build_roberta_lora
from training.optim import build_optimizer, build_scheduler
from training.trainer import Trainer
from utils.config import load_config
from utils.param_utils import print_trainable_parameters
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    model_type = cfg["model"]["type"]
    if model_type != "roberta":
        raise NotImplementedError(f"scripts/train.py currently supports roberta; got '{model_type}'")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    model, replaced = build_roberta_lora(
        model_name=cfg["model"]["name"],
        num_labels=cfg["task"]["num_labels"],
        rank=cfg["lora"]["rank"],
        alpha=cfg["lora"]["alpha"],
        dropout=cfg["lora"]["dropout"],
    )
    print(f"Injected LoRA into {len(replaced)} modules")
    print_trainable_parameters(model)

    data = load_glue(
        cfg["task"]["name"],
        tokenizer,
        max_length=cfg["task"]["max_length"],
    )
    train_ds = data["train"]
    eval_split = "validation_matched" if cfg["task"]["name"] == "mnli" else "validation"
    eval_ds = data[eval_split]

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

    def metrics_fn(preds, labels):
        return compute_glue_metrics(cfg["task"]["name"], preds.numpy(), labels.numpy())

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
    )
    trainer.train()
    print(f"best eval metric: {trainer.best_metric}")


if __name__ == "__main__":
    main()
