"""Distributed trainer for bilateral age regression using MONAI."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from monai.data import DataLoader, DistributedSampler
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from data.bilateral_dataset import create_datasets_from_csv
from models.bilateral_resnet_age import BilateralResNetAge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BilateralResNetAge with MONAI")
    parser.add_argument("--train-csv", type=str, required=True, help="CSV with training metadata")
    parser.add_argument("--val-csv", type=str, required=True, help="CSV with validation metadata")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Cache root for PersistentDataset")
    parser.add_argument("--output-dir", type=str, default="./runs", help="Directory to store checkpoints and logs")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50", "resnet101", "seresnet50", "seresnet101"], help="Encoder backbone")
    parser.add_argument("--feature-dim", type=int, default=256, help="Bottleneck dimension from encoder")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability in regressor")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for data loaders")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for AdamW")
    parser.add_argument(
        "--loss",
        type=str,
        default="mae",
        choices=["mae", "mse", "smooth_l1"],
        help="Regression loss to optimize",
    )
    parser.add_argument("--t-max", type=int, default=None, help="Epochs for a full cosine cycle; defaults to total epochs")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Lower bound for cosine annealing")
    parser.add_argument("--val-every", type=int, default=5, help="Run validation every N epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=20170111, help="Random seed for reproducibility")
    parser.add_argument("--log-every", type=int, default=20, help="Steps between logging training loss")
    parser.add_argument("--device", type=str, default="cuda", help="Preferred device (cuda or cpu) when not using DDP")
    parser.add_argument("--dist-backend", type=str, default="nccl", help="Backend for distributed training")
    parser.add_argument("--local-rank", dest="local_rank", type=int, default=-1, help="Local rank, set by torchrun for DDP")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Optional root directory appended to relative paths in the CSVs",
    )
    return parser.parse_args()


class BilateralAgeTrainer:
    """Class-style trainer that mirrors the distributed flow of the classification script."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model: Optional[torch.nn.Module] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.train_sampler: Optional[DistributedSampler] = None
        self.val_sampler: Optional[DistributedSampler] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[CosineAnnealingLR] = None
        self.loss_fn: Optional[torch.nn.Module] = None
        self.writer: Optional[SummaryWriter] = None
        self.scaler: Optional[GradScaler] = None
        self.run_dir: Optional[Path] = None
        self.device: Optional[torch.device] = None
        self.history: list[dict[str, Optional[float]]] = []
        self.best_val = float("inf")
        self.global_step = 0

        self.setup_distributed()
        self.setup_directories()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer_scheduler()
        self.setup_loss()
        self.setup_tracking()

    def setup_distributed(self) -> None:
        """Configure distributed backend and target device."""

        self.distributed = self.args.local_rank is not None and self.args.local_rank >= 0
        if self.distributed:
            dist.init_process_group(backend=self.args.dist_backend, init_method="env://")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device(f"cuda:{self.args.local_rank}")
        else:
            self.rank = 0
            self.world_size = 1
            prefer_cuda = self.args.device == "cuda" and torch.cuda.is_available()
            self.device = torch.device("cuda" if prefer_cuda else "cpu")
        self.is_main_process = self.rank == 0

    def setup_directories(self) -> None:
        """Create run directory and broadcast it to all ranks."""

        output_root = Path(self.args.output_dir)
        run_id = time.strftime("%Y%m%d-%H%M%S") if self.is_main_process else ""
        if self.is_main_process:
            output_root.mkdir(parents=True, exist_ok=True)

        if self.distributed:
            run_id = self._broadcast_string(run_id)

        self.run_dir = output_root / run_id
        if self.is_main_process:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        if self.distributed:
            dist.barrier()

    def setup_data(self) -> None:
        """Create datasets and their associated dataloaders."""

        train_ds, val_ds = create_datasets_from_csv(
            train_csv_path=self.args.train_csv,
            val_csv_path=self.args.val_csv,
            cache_dir=self.args.cache_dir,
            root_dir=self.args.root_dir,
        )

        if self.distributed:
            self.train_sampler = DistributedSampler(train_ds, even_divisible=True, shuffle=True)
            self.val_sampler = DistributedSampler(val_ds, even_divisible=True, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None

        pin_memory = self.device.type == "cuda"

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            num_workers=self.args.num_workers,
            pin_memory=pin_memory,
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=self.args.num_workers,
            pin_memory=pin_memory,
        )

    def setup_model(self) -> None:
        """Instantiate the bilateral model and wrap it for DDP when required."""

        self.model = BilateralResNetAge(
            backbone=self.args.backbone,
            feature_dim=self.args.feature_dim,
            dropout=self.args.dropout,
        ).to(self.device)

        if self.distributed:
            device_ids = [self.device.index] if self.device.type == "cuda" else None
            self.model = DistributedDataParallel(self.model, device_ids=device_ids)

        amp_enabled = self.args.amp and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=amp_enabled) if amp_enabled else None

    def setup_optimizer_scheduler(self) -> None:
        """Hook up optimizer and cosine scheduler."""

        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        t_max = self.args.t_max if self.args.t_max is not None else self.args.epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, t_max), eta_min=self.args.min_lr)

    def setup_loss(self) -> None:
        """Configure the regression loss function."""

        if self.args.loss == "mae":
            self.loss_fn = torch.nn.L1Loss()
        elif self.args.loss == "smooth_l1":
            self.loss_fn = torch.nn.SmoothL1Loss()
        else:
            self.loss_fn = torch.nn.MSELoss()

    def setup_tracking(self) -> None:
        """Initialize tensorboard writers and state trackers."""

        if self.is_main_process:
            self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        else:
            self.writer = None

    def train_epoch(self, epoch: int) -> float:
        """Run one epoch of training, returning the global mean loss."""

        assert self.train_loader is not None
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        sample_count = 0
        log_loss = 0.0
        log_count = 0
        total_steps = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, start=1):
            left = batch["LEFT"].to(self.device)
            right = batch["RIGHT"].to(self.device)
            age = batch["age"].to(self.device).float().view(-1)
            batch_size = age.shape[0]

            self.optimizer.zero_grad(set_to_none=True)
            amp_enabled = self.scaler is not None and self.scaler.is_enabled()
            with autocast(enabled=amp_enabled):
                preds = self.model(left, right)
                loss = self.loss_fn(preds, age)

            if amp_enabled:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item() * batch_size
            sample_count += batch_size
            log_loss += loss.item()
            log_count += 1

            if self.is_main_process:
                self.global_step += 1
                should_log = log_count == self.args.log_every or step == total_steps
                if should_log:
                    avg = log_loss / log_count
                    print(
                        f"Epoch {epoch + 1}/{self.args.epochs} Step {step}/{total_steps} | train_loss={avg:.4f}"
                    )
                    if self.writer is not None:
                        self.writer.add_scalar("loss/train_iter", avg, self.global_step)
                    log_loss = 0.0
                    log_count = 0

        loss_tensor = torch.tensor([epoch_loss], device=self.device)
        count_tensor = torch.tensor([sample_count], device=self.device)
        if self.distributed:
            dist.all_reduce(loss_tensor)
            dist.all_reduce(count_tensor)

        total_loss = loss_tensor.item()
        total_count = max(int(count_tensor.item()), 1)
        avg_loss = total_loss / total_count

        return avg_loss

    def validate_epoch(self, epoch: int) -> float:
        """Evaluate on the validation set and return the reduced mean loss."""

        assert self.val_loader is not None
        self.model.eval()
        if self.val_sampler is not None:
            self.val_sampler.set_epoch(epoch)

        total_loss = torch.tensor([0.0], device=self.device)
        total_count = torch.tensor([0.0], device=self.device)

        with torch.no_grad():
            for batch in self.val_loader:
                left = batch["LEFT"].to(self.device)
                right = batch["RIGHT"].to(self.device)
                age = batch["age"].to(self.device).float().view(-1)
                batch_size = age.shape[0]

                preds = self.model(left, right)
                loss = self.loss_fn(preds, age)
                total_loss += loss.item() * batch_size
                total_count += batch_size

        if self.distributed:
            dist.all_reduce(total_loss)
            dist.all_reduce(total_count)

        mean_loss = total_loss.item() / max(total_count.item(), 1.0)
        if self.writer is not None and self.is_main_process:
            self.writer.add_scalar("loss/val", mean_loss, epoch + 1)
        return mean_loss

    def maybe_update_best(self, val_loss: Optional[float]) -> None:
        if val_loss is None or not self.is_main_process or self.run_dir is None:
            return
        if val_loss < self.best_val:
            self.best_val = val_loss
            torch.save(self._model_state_dict(), self.run_dir / "model_best.pt")
            print("Saved new best model.")

    def record_history(self, epoch: int, lr: float, val_loss: Optional[float]) -> None:
        if not self.is_main_process:
            return
        self.history.append({"epoch": epoch + 1, "val_loss": val_loss, "lr": lr})

    def save_checkpoint(self, epoch: int, val_loss: Optional[float]) -> None:
        if not self.is_main_process or self.run_dir is None:
            return
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": self._model_state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, self.run_dir / "checkpoint_last.pt")

    def train(self) -> None:
        """Main training loop with optional validation intervals."""

        for epoch in range(self.args.epochs):
            if self.is_main_process:
                print(f"\n{'=' * 50}")
                print(f"Epoch {epoch + 1}/{self.args.epochs}")
                print(f"{'=' * 50}")

            train_loss = self.train_epoch(epoch)
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            if self.writer is not None and self.is_main_process:
                self.writer.add_scalar("lr", lr, epoch + 1)
                self.writer.add_scalar("loss/train_epoch", train_loss, epoch + 1)

            if self.is_main_process:
                print(f"Epoch {epoch + 1} | train_loss={train_loss:.4f} | lr={lr:.2e}")

            should_validate = (epoch + 1) % self.args.val_every == 0 or (epoch + 1) == self.args.epochs
            val_loss = None
            if should_validate:
                val_loss = self.validate_epoch(epoch)
                if self.is_main_process:
                    print(f"Epoch {epoch + 1} | val_loss={val_loss:.4f}")
            else:
                if self.is_main_process:
                    print(f"Epoch {epoch + 1} | validation skipped")

            self.maybe_update_best(val_loss)
            self.record_history(epoch, lr, val_loss)
            self.save_checkpoint(epoch, val_loss)

        self.finalize()

    def finalize(self) -> None:
        """Persist history and tear down distributed resources."""

        if self.writer is not None:
            self.writer.close()

        if self.is_main_process and self.run_dir is not None:
            history_path = self.run_dir / "history.json"
            history_path.write_text(json.dumps(self.history, indent=2))
            print(f"Training complete. Logs and checkpoints in {self.run_dir}")

        if self.distributed:
            dist.barrier()
            dist.destroy_process_group()

    def _broadcast_string(self, value: str) -> str:
        if not self.distributed:
            return value
        obj_list = [value]
        dist.broadcast_object_list(obj_list, src=0)
        return obj_list[0]

    def _model_state_dict(self) -> dict:
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()


def main() -> None:
    args = parse_args()
    if args.local_rank < 0:
        env_rank = os.environ.get("LOCAL_RANK")
        if env_rank is not None:
            args.local_rank = int(env_rank)
    set_determinism(seed=args.seed)
    trainer = BilateralAgeTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
