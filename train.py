"""
Training pipeline for CBCT tooth & root canal segmentation.

Usage:
    python train.py --arch swin_unetr --epochs 300 --batch_size 2
    python train.py --arch unet3d --two_stage
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from config import AugmentConfig, DataConfig, ModelConfig, TrainConfig
from dataset import get_dataloaders
from losses import CombinedLoss, DeepSupervisionLoss
from model import build_model
from transforms import get_train_transforms, get_val_transforms


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
) -> dict:
    """Compute per-class Dice scores."""
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    # One-hot encode
    preds_one_hot = torch.nn.functional.one_hot(preds.long(), num_classes)
    preds_one_hot = preds_one_hot.permute(0, 4, 1, 2, 3)
    targets_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes)
    targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3)

    dice_metric(preds_one_hot, targets_one_hot)
    dice_values = dice_metric.aggregate()
    dice_metric.reset()

    return {
        "dice_tooth": dice_values[0].item(),
        "dice_canal": dice_values[1].item(),
        "dice_mean": dice_values.mean().item(),
    }


class Trainer:
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        train_config: TrainConfig,
        aug_config: AugmentConfig,
    ):
        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config
        self.aug_config = aug_config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Build model
        self.model = build_model(model_config, img_size=data_config.patch_size)
        self.model = self.model.to(self.device)

        # Loss
        base_loss = CombinedLoss(
            num_classes=data_config.num_classes,
            dice_weight=train_config.dice_weight,
            focal_weight=train_config.focal_weight,
            focal_gamma=train_config.focal_gamma,
            focal_alpha=train_config.focal_alpha,
        )
        if model_config.architecture == "unet3d":
            self.criterion = DeepSupervisionLoss(base_loss)
        else:
            self.criterion = base_loss

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        # Scheduler
        if train_config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=train_config.epochs, T_mult=1
            )
        else:  # polynomial
            self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
                self.optimizer, total_iters=train_config.epochs, power=0.9
            )

        # Warmup scheduler
        if train_config.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                total_iters=train_config.warmup_epochs,
            )

        # AMP scaler
        self.scaler = GradScaler(enabled=train_config.use_amp)
        self.use_amp = train_config.use_amp

        # Dataloaders
        train_transforms = get_train_transforms(data_config, aug_config)
        val_transforms = get_val_transforms(data_config)
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            data_config,
            train_transforms,
            val_transforms,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            oversample_canal=train_config.oversample_canal,
            canal_oversample_ratio=train_config.canal_oversample_ratio,
            group_by_case=data_config.group_by_case,
        )

        # Logging
        self.writer = SummaryWriter(log_dir=os.path.join(
            train_config.log_dir, train_config.experiment_name
        ))
        Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.best_val_dice = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        epoch_loss = 0.0
        step_count = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                output = self.model(images)

                # Handle deep supervision (UNet3D in training mode)
                if isinstance(output, tuple):
                    main_out, deep_outs = output
                    loss = self.criterion(main_out, deep_outs, labels)
                else:
                    loss = self.criterion(output, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            step_count += 1

        # Step schedulers
        if epoch < self.train_config.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.scheduler.step()

        return {"train_loss": epoch_loss / max(step_count, 1)}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        self.model.eval()
        val_loss = 0.0
        all_dice_tooth = []
        all_dice_canal = []

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Use sliding window inference for full volumes
            output = sliding_window_inference(
                images,
                roi_size=self.data_config.patch_size,
                sw_batch_size=4,
                predictor=self.model,
                overlap=0.5,
            )

            if isinstance(self.criterion, DeepSupervisionLoss):
                loss = self.criterion.base_loss(output, labels)
            else:
                loss = self.criterion(output, labels)
            val_loss += loss.item()

            # Compute dice
            preds = output.argmax(dim=1)
            if labels.dim() == 5:
                labels = labels.squeeze(1)
            metrics = compute_metrics(preds, labels, self.data_config.num_classes)
            all_dice_tooth.append(metrics["dice_tooth"])
            all_dice_canal.append(metrics["dice_canal"])

        n = max(len(self.val_loader), 1)
        return {
            "val_loss": val_loss / n,
            "val_dice_tooth": np.mean(all_dice_tooth),
            "val_dice_canal": np.mean(all_dice_canal),
            "val_dice_mean": np.mean(all_dice_tooth + all_dice_canal) / 2
            if all_dice_tooth
            else 0.0,
        }

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_dice": self.best_val_dice,
            "metrics": metrics,
            "model_config": self.model_config,
            "data_config": self.data_config,
        }
        ckpt_dir = self.train_config.checkpoint_dir

        if is_best:
            torch.save(state, os.path.join(ckpt_dir, "best_model.pth"))
            print(f"  -> Saved best model (dice={metrics.get('val_dice_mean', 0):.4f})")

        if (epoch + 1) % self.train_config.save_every == 0:
            torch.save(state, os.path.join(ckpt_dir, f"checkpoint_epoch{epoch+1}.pth"))

    def train(self):
        print(f"\nStarting training for {self.train_config.epochs} epochs...")
        print(f"  Architecture: {self.model_config.architecture}")
        print(f"  Patch size: {self.data_config.patch_size}")
        print(f"  Batch size: {self.train_config.batch_size}")
        print(f"  Loss: Dice(w={self.train_config.dice_weight}) + Focal(w={self.train_config.focal_weight})")
        print(f"  Canal oversampling: {self.train_config.oversample_canal}")
        print()

        for epoch in range(self.train_config.epochs):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            # Log to tensorboard
            self.writer.add_scalar("Loss/train", train_metrics["train_loss"], epoch)
            self.writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
            self.writer.add_scalar("Dice/tooth", val_metrics["val_dice_tooth"], epoch)
            self.writer.add_scalar("Dice/canal", val_metrics["val_dice_canal"], epoch)
            self.writer.add_scalar("Dice/mean", val_metrics["val_dice_mean"], epoch)
            self.writer.add_scalar("LR", lr, epoch)

            # Print progress
            print(
                f"Epoch [{epoch+1}/{self.train_config.epochs}] "
                f"({elapsed:.1f}s) lr={lr:.2e} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"dice_tooth={val_metrics['val_dice_tooth']:.4f} | "
                f"dice_canal={val_metrics['val_dice_canal']:.4f}"
            )

            # Check for improvement
            mean_dice = (val_metrics["val_dice_tooth"] + val_metrics["val_dice_canal"]) / 2
            is_best = mean_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = mean_dice
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best)

            # Early stopping
            if self.patience_counter >= self.train_config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {self.patience_counter} epochs)")
                break

        self.writer.close()
        print(f"\nTraining complete. Best val dice: {self.best_val_dice:.4f}")
        print(f"Best model saved to: {self.train_config.checkpoint_dir}/best_model.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CBCT Tooth & Canal Segmentation")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--arch", type=str, default="swin_unetr", choices=["unet3d", "swin_unetr"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_oversample", action="store_true")
    parser.add_argument("--two_stage", action="store_true")
    parser.add_argument("--experiment", type=str, default="cbct_tooth_canal_seg")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    data_config = DataConfig(
        data_dir=args.data_dir,
        patch_size=tuple(args.patch_size),
    )
    model_config = ModelConfig(architecture=args.arch)
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        oversample_canal=not args.no_oversample,
        two_stage=args.two_stage,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment,
    )
    aug_config = AugmentConfig()

    trainer = Trainer(data_config, model_config, train_config, aug_config)

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.best_val_dice = ckpt.get("best_val_dice", 0.0)

    trainer.train()


if __name__ == "__main__":
    main()
