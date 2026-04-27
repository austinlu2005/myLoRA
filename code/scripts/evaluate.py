import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# =========================
# LoRA Layer
# =========================
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()

        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # freeze original weights
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


# =========================
# Utils
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =========================
# Inject LoRA into CLIP
# =========================
def inject_lora(model, r=8, alpha=16):
    replaced = []

    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if child_name in ["q_proj", "v_proj"] and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
                replaced.append(f"{name}.{child_name}")

    print(f"Injected LoRA into {len(replaced)} layers")
    return replaced


# =========================
# Dataset
# =========================
def find_columns(dataset):
    cols = dataset.column_names

    image_col = None
    text_col = None

    for c in cols:
        if "image" in c:
            image_col = c
        if "caption" in c or "text" in c:
            text_col = c

    return image_col, text_col


def make_collate_fn(processor, image_col, text_col):
    def collate_fn(batch):
        images = []
        texts = []

        for ex in batch:
            img = ex[image_col]
            if not isinstance(img, Image.Image):
                img = Image.open(img)

            img = img.convert("RGB")
            text = ex[text_col]

            if isinstance(text, list):
                text = text[0]

            images.append(img)
            texts.append(text)

        enc = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        return enc

    return collate_fn


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    img_embeds = []
    txt_embeds = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)

        img = outputs.image_embeds
        txt = outputs.text_embeds

        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)

        img_embeds.append(img.cpu())
        txt_embeds.append(txt.cpu())

    img_embeds = torch.cat(img_embeds)
    txt_embeds = torch.cat(txt_embeds)

    sims = img_embeds @ txt_embeds.T
    labels = torch.arange(len(sims))

    def recall_at_k(sim, k):
        topk = sim.topk(k, dim=1).indices
        return (topk == labels[:, None]).any(dim=1).float().mean().item()

    return {
        "R@1": recall_at_k(sims, 1),
        "R@5": recall_at_k(sims, min(5, sims.size(1)))
    }


# =========================
# Training
# =========================
def train(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model)

    freeze_all(model)

    if args.method == "lora":
        inject_lora(model, args.rank, args.alpha)

    model.to(device)

    total, trainable = count_params(model)
    print(f"Trainable params: {trainable} / {total}")

    dataset = load_dataset(args.dataset, split="train")
    dataset = dataset.shuffle().select(range(args.max_samples))

    image_col, text_col = find_columns(dataset)

    split = int(0.8 * len(dataset))
    train_ds = dataset.select(range(split))
    val_ds = dataset.select(range(split, len(dataset)))

    collate_fn = make_collate_fn(processor, image_col, text_col)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch, return_loss=True)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"\nEpoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

        metrics = evaluate(model, val_loader, device)
        print(metrics)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", default="lora", choices=["lora", "frozen"])
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dataset", default="lambdalabs/pokemon-blip-captions")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)

    parser.add_argument("--max_samples", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()