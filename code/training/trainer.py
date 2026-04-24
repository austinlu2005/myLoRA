from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        optimizer,
        scheduler,
        batch_size,
        num_epochs,
        device,
        compute_metrics=None,
        log_steps=50,
        grad_clip=1.0,
        output_dir=None,
    ):
        self.model = model.to(device)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        self.eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.compute_metrics = compute_metrics
        self.log_steps = log_steps
        self.grad_clip = grad_clip
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
            out = self.model(**batch)
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
        all_preds = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0
        for batch in self.eval_loader:
            batch = self._move_to_device(batch)
            out = self.model(**batch)
            total_loss += out.loss.item()
            n_batches += 1
            preds = out.logits.argmax(dim=-1)
            all_preds.append(preds.detach().cpu())
            all_labels.append(batch["labels"].detach().cpu())
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        metrics = {"eval_loss": total_loss / max(n_batches, 1)}
        if self.compute_metrics is not None:
            metrics.update(self.compute_metrics(preds, labels))
        return metrics

    def _maybe_save_best(self, metrics):
        if self.output_dir is None:
            return
        score_key = next(
            (k for k in ("accuracy", "f1", "matthews_correlation", "pearson") if k in metrics),
            None,
        )
        if score_key is None:
            return
        score = metrics[score_key]
        if self.best_metric is None or score > self.best_metric:
            self.best_metric = score
            from lora.save_load import save_lora_state_dict

            save_lora_state_dict(self.model, self.output_dir / "lora_best.pt")
            torch.save(
                {k: v for k, v in self.model.state_dict().items() if "classifier" in k},
                self.output_dir / "classifier_best.pt",
            )
