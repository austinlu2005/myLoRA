from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evaluation.vlm_metrics import compute_clip_retrieval_metrics
from lora.save_load import save_lora_state_dict


class CLIPRetrievalTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        collate_fn,
        optimizer,
        scheduler,
        batch_size,
        num_epochs,
        device,
        grad_clip=1.0,
        log_steps=10,
        output_dir=None,
        primary_metric="mean_recall@1",
        ks=(1, 5),
    ):
        self.model = model.to(device)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.grad_clip = grad_clip
        self.log_steps = log_steps
        self.primary_metric = primary_metric
        self.ks = tuple(ks)
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_metric = None
        self.history = []

    def _move_to_device(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def train(self):
        for epoch in range(self.num_epochs):
            self._train_one_epoch(epoch)
            metrics = self.evaluate()
            self.history.append({"epoch": epoch, **metrics})
            print(f"[epoch {epoch}] eval: {metrics}")
            self._maybe_save_best(metrics)
        return self.history

    def _train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_n = 0
        pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            batch = self._move_to_device(batch)
            out = self.model(**batch, return_loss=True)
            loss = out.loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad),
                    self.grad_clip,
                )
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()
            running_n += 1
            self.global_step += 1

            if self.global_step % self.log_steps == 0:
                avg = running_loss / max(running_n, 1)
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")
                running_loss = 0.0
                running_n = 0

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_image_embeds = []
        all_text_embeds = []

        for batch in self.eval_loader:
            batch = self._move_to_device(batch)
            out = self.model(**batch, return_loss=True)
            total_loss += out.loss.item()
            n_batches += 1
            all_image_embeds.append(out.image_embeds.detach().cpu())
            all_text_embeds.append(out.text_embeds.detach().cpu())

        image_embeds = torch.cat(all_image_embeds, dim=0)
        text_embeds = torch.cat(all_text_embeds, dim=0)

        metrics = {"eval_loss": total_loss / max(n_batches, 1)}
        metrics.update(compute_clip_retrieval_metrics(image_embeds, text_embeds, ks=self.ks))
        return metrics

    def _maybe_save_best(self, metrics):
        score = metrics[self.primary_metric]
        if self.best_metric is None or score > self.best_metric:
            self.best_metric = score
            if self.output_dir is not None:
                save_lora_state_dict(self.model, self.output_dir / "lora_best.pt")
