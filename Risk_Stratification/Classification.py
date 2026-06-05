"""
Public training script for 3D ResNet Unilateral Classification.
Prepared for peer review and open-source release.
"""

import argparse
import os
import random
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.data import PersistentDataset
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from data.unilateral_dataset import (
    _build_transforms,
    _prepare_records,
    _validate_dataframe,
    create_unilateral_datasets,
)
from models import UnilateralResNetCls


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics_with_ci(y_true, y_probs, n_bootstraps=1000, alpha=0.95):
    """Compute AUC, Sensitivity, and Specificity with Bootstrap Confidence Intervals."""
    y_pred = np.argmax(y_probs, axis=1) if y_probs.ndim > 1 else (y_probs > 0.5).astype(int)
    pos_probs = y_probs[:, 1] if y_probs.ndim > 1 else y_probs

    try:
        original_auc = roc_auc_score(y_true, pos_probs)
    except ValueError:
        original_auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    original_sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    original_spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    aucs, sens, spes = [], [], []
    rng = np.random.RandomState(42)
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue

        y_true_boot = y_true[indices]
        y_probs_boot = pos_probs[indices]
        y_pred_boot = y_pred[indices]

        try:
            aucs.append(roc_auc_score(y_true_boot, y_probs_boot))
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_boot, y_pred_boot).ravel()
            sens.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0)
            spes.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0)
        except ValueError:
            continue

    def get_ci(values):
        if not values:
            return 0.0, 0.0
        lower = np.percentile(values, (1 - alpha) / 2 * 100)
        upper = np.percentile(values, (alpha + (1 - alpha) / 2) * 100)
        return lower, upper

    return (original_auc, get_ci(aucs)), (original_sen, get_ci(sens)), (original_spe, get_ci(spes))


def parse_args():
    parser = argparse.ArgumentParser(description="Unilateral 3D ResNet Training Pipeline")
    parser.add_argument("--train-csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--val-csv", type=str, required=True, help="Path to internal validation CSV")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to internal test dataset")
    parser.add_argument("--external-csv1", type=str, default=None, help="Path to external test dataset 1")
    parser.add_argument("--external-csv2", type=str, default=None, help="Path to external test dataset 2")
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, default="./runs_uni")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--root-dir", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training and run evaluation only")
    parser.add_argument("--eval-prefix", type=str, default="eval", help="Prefix for output evaluation CSV")

    args = parser.parse_args()

    if args.evaluate_only:
        args.output_dir = f"{args.output_dir}_eval"
    elif args.pretrained:
        args.output_dir = f"{args.output_dir}_pretrained"
    else:
        args.output_dir = f"{args.output_dir}_scratch"

    return args


class UnilateralTrainer:
    def __init__(self, args):
        self.args = args
        self.best_auc = 0.0

        # Distributed Training Setup
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.local_rank = env_local_rank if env_local_rank >= 0 else args.local_rank
        self.distributed = self.local_rank >= 0

        if self.distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.rank = dist.get_rank()
            self.is_main = self.rank == 0
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.rank = 0
            self.is_main = True

        self._setup_data()
        self._setup_model()

        if self.is_main:
            os.makedirs(args.output_dir, exist_ok=True)
            if not self.args.evaluate_only:
                self.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))

    def _setup_data(self):
        datasets = create_unilateral_datasets(
            train_csv_path=self.args.train_csv,
            val_csv_path=self.args.val_csv,
            test_csv_path=self.args.test_csv,
            cache_dir=self.args.cache_dir,
            root_dir=self.args.root_dir
        )

        train_ds = datasets[0]
        val_ds = datasets[1]

        train_sampler = DistributedSampler(train_ds, shuffle=True) if self.distributed else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if self.distributed else None

        self.train_loader = DataLoader(
            train_ds, batch_size=self.args.batch_size,
            sampler=train_sampler, shuffle=(train_sampler is None),
            num_workers=self.args.num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.args.batch_size,
            sampler=val_sampler, shuffle=False,
            num_workers=self.args.num_workers, pin_memory=True
        )

        # Optional test sets setup
        self.test_loader = self._create_test_loader(datasets[2] if len(datasets) == 3 else None)

        transforms = _build_transforms((256, 256, 64))
        root_path = Path(self.args.root_dir).expanduser() if self.args.root_dir else None

        self.ext_loader1 = self._create_external_loader(self.args.external_csv1, root_path, transforms, "ext1")
        self.ext_loader2 = self._create_external_loader(self.args.external_csv2, root_path, transforms, "ext2")

    def _create_test_loader(self, dataset):
        if dataset is None:
            return None
        sampler = DistributedSampler(dataset, shuffle=False) if self.distributed else None
        return DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler,
                          shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

    def _create_external_loader(self, csv_path, root_path, transforms, cache_prefix):
        if not csv_path:
            return None
        df = _validate_dataframe(Path(csv_path))
        records = _prepare_records(df, root_path, validate_files=True)
        cache_dir = Path(self.args.cache_dir) / f"{cache_prefix}_uni"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ds = PersistentDataset(data=records, transform=transforms, cache_dir=str(cache_dir))
        sampler = DistributedSampler(ds, shuffle=False) if self.distributed else None
        return DataLoader(ds, batch_size=self.args.batch_size, sampler=sampler,
                          shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

    def _setup_model(self):
        self.model = UnilateralResNetCls(backbone=self.args.backbone).to(self.device)

        if self.args.pretrained and os.path.isfile(self.args.pretrained):
            if self.is_main:
                print(f"[*] Loading pretrained weights from: {self.args.pretrained}")
            checkpoint = torch.load(self.args.pretrained, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
            model_dict = self.model.state_dict()
            pretrained_dict = {
                k.replace("module.", ""): v
                for k, v in state_dict.items()
                if k.replace("module.", "") in model_dict and model_dict[k.replace("module.", "")].shape == v.shape
            }
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

        if self.distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                 output_device=self.local_rank)

        self.criterion = torch.nn.CrossEntropyLoss()
        if not self.args.evaluate_only:
            self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=self.args.min_lr)
            self.scaler = torch.amp.GradScaler('cuda')

    def _gather_tensor(self, tensor):
        if not self.distributed: return tensor
        gathered_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_list, tensor)
        return torch.cat(gathered_list, dim=0)

    def train_epoch(self, epoch):
        self.model.train()
        if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss, total_acc, total_num = 0.0, 0.0, 0

        for batch in self.train_loader:
            img = batch["image"].to(self.device)
            label = batch["label"].to(self.device).long().view(-1)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = self.model(img)
                loss = self.criterion(logits, label)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            total_acc += (logits.argmax(dim=1) == label).float().sum().item()
            total_num += batch_size

        stats = torch.tensor([total_loss, total_acc, total_num], device=self.device)
        if self.distributed: dist.all_reduce(stats)

        return stats[0].item() / stats[2].item(), stats[1].item() / stats[2].item()

    @torch.no_grad()
    def validate(self, epoch_str, loader, prefix="val", calculate_ci=False):
        self.model.eval()
        total_loss, total_acc, total_num = 0.0, 0.0, 0
        val_probs_list, val_labels_list = [], []

        for batch in loader:
            img = batch["image"].to(self.device)
            label = batch["label"].to(self.device).long().view(-1)
            batch_size = label.size(0)

            logits = self.model(img)
            loss = self.criterion(logits, label)

            total_loss += loss.item() * batch_size
            total_acc += (logits.argmax(dim=1) == label).float().sum().item()
            total_num += batch_size

            val_probs_list.append(F.softmax(logits, dim=1))
            val_labels_list.append(label)

        stats = torch.tensor([total_loss, total_acc, total_num], device=self.device)
        if self.distributed: dist.all_reduce(stats)
        avg_loss = stats[0].item() / stats[2].item()
        avg_acc = stats[1].item() / stats[2].item()

        auc, csv_path = 0.0, None

        local_probs = torch.cat(val_probs_list, dim=0)
        local_labels = torch.cat(val_labels_list, dim=0)
        global_probs = self._gather_tensor(local_probs)
        global_labels = self._gather_tensor(local_labels)

        if self.is_main:
            y_true = global_labels.cpu().numpy()
            y_scores = global_probs.cpu().numpy()
            num_classes = y_scores.shape[1]
            y_pred_pos = y_scores[:, 1] if num_classes > 1 else y_scores
            y_pred_class = np.argmax(y_scores, axis=1) if num_classes > 1 else (y_scores > 0.5).astype(int)

            df_results = pd.DataFrame({
                'y_true': y_true,
                'y_prob_positive': y_pred_pos,
                'y_pred_class': y_pred_class
            })

            file_name = f"{prefix}_preds_epoch_{epoch_str}.csv"
            csv_path = os.path.join(self.args.output_dir, file_name)
            df_results.to_csv(csv_path, index=False)

            if num_classes == 2:
                if calculate_ci:
                    (auc, ci), (sen, _), (spe, _) = compute_metrics_with_ci(y_true, y_scores)
                    print(
                        f"[{prefix.upper()} | Ep {epoch_str}] AUC: {auc:.4f} (95% CI: {ci[0]:.4f}-{ci[1]:.4f}) | Sen: {sen:.4f} Spe: {spe:.4f}")
                else:
                    try:
                        auc = roc_auc_score(y_true, y_pred_pos)
                    except ValueError:
                        auc = 0.0
                    print(
                        f"[{prefix.upper()} | Ep {epoch_str}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | AUC: {auc:.4f}")
            else:
                auc = roc_auc_score(y_true, y_scores, multi_class='ovr')
                print(f"[{prefix.upper()}] Epoch {epoch_str} | AUC: {auc:.4f}")

        return avg_loss, avg_acc, auc, csv_path

    def run(self):
        if self.args.evaluate_only:
            if self.is_main:
                print(f"========== EXTERNAL VALIDATION MODE ({self.args.eval_prefix}) ==========")
            self.validate(epoch_str="eval", loader=self.val_loader, prefix=self.args.eval_prefix, calculate_ci=True)
            if self.distributed: dist.destroy_process_group()
            return

        if self.is_main:
            print("========== STARTING TRAINING ==========")

        for epoch in range(1, self.args.epochs + 1):
            t_loss, t_acc = self.train_epoch(epoch)
            self.scheduler.step()

            if self.is_main:
                print(f"Epoch {epoch}/{self.args.epochs} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f}")
                self.writer.add_scalar("train/loss", t_loss, epoch)
                self.writer.add_scalar("train/acc", t_acc, epoch)

            if epoch % 2 == 0 or epoch == self.args.epochs:
                v_loss, v_acc, v_auc, current_csv_path = self.validate(
                    epoch_str=epoch, loader=self.val_loader, prefix="val", calculate_ci=False
                )

                if self.is_main:
                    self.writer.add_scalar("val/loss", v_loss, epoch)
                    self.writer.add_scalar("val/auc", v_auc, epoch)

                    if v_auc > self.best_auc:
                        self.best_auc = v_auc
                        save_path = os.path.join(self.args.output_dir, "best_model.pt")
                        torch.save(self.model.state_dict(), save_path)
                        print(f" >>> [SAVE] New Best Val AUC: {v_auc:.4f} saved to {save_path}")
                        if current_csv_path and os.path.exists(current_csv_path):
                            shutil.copy(current_csv_path, os.path.join(self.args.output_dir, "best_val_preds.csv"))

        if self.is_main:
            print("\n===================================================================")
            print("      TRAINING COMPLETE. RUNNING FINAL EVALUATION ON BEST MODEL    ")
            print("===================================================================")

        best_model_path = os.path.join(self.args.output_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=False))

            if self.test_loader:
                self.validate("best", self.test_loader, "final_int_test", calculate_ci=True)
            if self.ext_loader1:
                self.validate("best", self.ext_loader1, "final_ext1", calculate_ci=True)
            if self.ext_loader2:
                self.validate("best", self.ext_loader2, "final_ext2", calculate_ci=True)

        if self.is_main:
            self.writer.close()
            print(f"\nAll Tasks Finished. Artifacts saved in: {os.path.abspath(self.args.output_dir)}")

        if self.distributed: dist.destroy_process_group()


if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    UnilateralTrainer(args).run()