import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.vlm import load_clip_retrieval_data, make_clip_collate_fn
from models.vlm_wrapper import build_clip_lora
from training.optim import build_optimizer, build_scheduler
from training.vlm_trainer import CLIPRetrievalTrainer
from utils.config import load_config
from utils.param_utils import print_trainable_parameters
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    processor = AutoProcessor.from_pretrained(cfg["model"]["name"])
    model, replaced = build_clip_lora(
        model_name=cfg["model"]["name"],
        rank=cfg["lora"]["rank"],
        alpha=cfg["lora"]["alpha"],
        dropout=cfg["lora"]["dropout"],
        tower=cfg["model"].get("tower", "both"),
    )

    data = load_clip_retrieval_data(
        dataset_name=cfg["task"]["dataset"],
        split=cfg["task"].get("split", "train"),
        max_samples=cfg["task"].get("max_samples"),
        eval_ratio=cfg["task"].get("eval_ratio", 0.2),
        seed=cfg["training"]["seed"],
    )
    collate_fn = make_clip_collate_fn(
        processor,
        image_col=data["image_col"],
        text_col=data["text_col"],
        max_length=cfg["task"].get("max_text_length"),
    )

    print(f"Injected LoRA into {len(replaced)} modules")
    print_trainable_parameters(model)

    train_ds = data["train"]
    eval_ds = data["validation"]
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
        warmup_steps=cfg["training"].get("warmup_steps"),
        warmup_ratio=cfg["training"].get("warmup_ratio"),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = CLIPRetrievalTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        collate_fn=collate_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=cfg["training"]["batch_size"],
        num_epochs=cfg["training"]["num_epochs"],
        device=device,
        grad_clip=cfg["training"].get("grad_clip", 1.0),
        log_steps=cfg["training"].get("log_steps", 10),
        output_dir=cfg.get("output", {}).get("dir"),
        primary_metric=cfg["task"].get("primary_metric", "mean_recall@1"),
        ks=tuple(cfg["task"].get("recall_at", [1, 5])),
    )

    t0 = time.time()
    history = trainer.train()
    wall = time.time() - t0

    result = {
        "task": cfg["task"].get("name", "image_text_retrieval"),
        "dataset": cfg["task"]["dataset"],
        "image_column": data["image_col"],
        "text_column": data["text_col"],
        "primary_metric": trainer.primary_metric,
        "best_metric": trainer.best_metric,
        "wall_seconds": wall,
        "history": history,
        "hparams": cfg["training"],
        "lora": cfg["lora"],
        "model": cfg["model"],
        "replaced_modules": replaced,
    }

    output_dir = Path(cfg.get("output", {}).get("dir", "./runs/clip_lora"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(f"best {trainer.primary_metric}: {trainer.best_metric}")
    print(f"wall minutes: {wall / 60:.1f}")
    print(f"saved result to {output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
