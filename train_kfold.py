"""
K-fold cross-validation training ở mức case.

Với 4 ca CBCT × 6 răng = 24 mẫu, chạy 4-fold CV:
    - Fold 1: train trên ca [1,2,3], val trên ca [4]
    - Fold 2: train trên ca [1,2,4], val trên ca [3]
    - Fold 3: train trên ca [1,3,4], val trên ca [2]
    - Fold 4: train trên ca [2,3,4], val trên ca [1]
    -> mỗi ca được validate đúng 1 lần, tận dụng 100% dữ liệu.

Cách dùng:
    python train_kfold.py --data_dir data/teeth --arch swin_unetr --n_folds 4 --epochs 200

    # Chạy lại 1 fold cụ thể:
    python train_kfold.py --data_dir data/teeth --only_fold 2

    # Train nhanh để debug:
    python train_kfold.py --data_dir data/teeth --epochs 10 --n_folds 2
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from config import AugmentConfig, DataConfig, ModelConfig, TrainConfig
from dataset import (
    get_fold_dataloaders,
    kfold_split_by_case,
    prepare_data_list,
)
from train import Trainer
from transforms import get_train_transforms, get_val_transforms


class KFoldTrainer(Trainer):
    """
    Trainer con, nhận sẵn train_loader/val_loader thay vì tự tạo
    từ DataConfig (để tái sử dụng split của K-fold).
    """

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        train_config: TrainConfig,
        aug_config: AugmentConfig,
        train_loader,
        val_loader,
    ):
        # Gọi super nhưng không muốn nó tự tạo dataloaders.
        # Cách đơn giản: copy phần init quan trọng, bỏ qua phần load data.
        from losses import CombinedLoss, DeepSupervisionLoss
        from model import build_model
        from torch.cuda.amp import GradScaler
        from torch.utils.tensorboard import SummaryWriter

        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config
        self.aug_config = aug_config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = build_model(model_config, img_size=data_config.patch_size)
        self.model = self.model.to(self.device)

        base_loss = CombinedLoss(
            num_classes=data_config.num_classes,
            dice_weight=train_config.dice_weight,
            focal_weight=train_config.focal_weight,
            focal_gamma=train_config.focal_gamma,
            class_weights=train_config.class_weights,
        )
        if model_config.architecture in ("unet3d", "nnunet"):
            self.criterion = DeepSupervisionLoss(base_loss)
        else:
            self.criterion = base_loss

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        if train_config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=train_config.epochs, T_mult=1
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
                self.optimizer, total_iters=train_config.epochs, power=0.9
            )

        if train_config.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                total_iters=train_config.warmup_epochs,
            )

        self.scaler = GradScaler(enabled=train_config.use_amp)
        self.use_amp = train_config.use_amp

        # Dùng dataloaders được truyền vào, bỏ qua test_loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = None

        self.writer = SummaryWriter(log_dir=os.path.join(
            train_config.log_dir, train_config.experiment_name
        ))
        Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.best_val_dice = 0.0
        self.patience_counter = 0


def run_kfold(args):
    """Chạy toàn bộ K-fold CV và tổng hợp kết quả."""
    # Chuẩn bị config chung
    data_config = DataConfig(
        data_dir=args.data_dir,
        patch_size=tuple(args.patch_size),
    )
    model_config = ModelConfig(architecture=args.arch)
    aug_config = AugmentConfig()

    # Tạo transforms 1 lần, dùng chung cho mọi fold
    train_transforms = get_train_transforms(data_config, aug_config)
    val_transforms = get_val_transforms(data_config)

    # Chia K-fold ở mức case
    data_list = prepare_data_list(args.data_dir)
    print(f"\nTổng số răng: {len(data_list)}")

    folds = kfold_split_by_case(
        data_list, n_folds=args.n_folds, seed=data_config.seed
    )
    n_folds = len(folds)

    # Thư mục output cho toàn bộ K-fold experiment
    kfold_root = Path(args.checkpoint_dir) / args.experiment
    kfold_root.mkdir(parents=True, exist_ok=True)

    # Chạy từng fold
    fold_results = []
    t_start = time.time()

    for fold_idx, (train_data, val_data) in enumerate(folds):
        fold_num = fold_idx + 1

        # Nếu user chỉ định 1 fold, skip các fold khác
        if args.only_fold is not None and fold_num != args.only_fold:
            continue

        print(f"\n{'='*70}")
        print(f"FOLD {fold_num}/{n_folds}")
        print(f"  Train: {len(train_data)} răng")
        print(f"  Val:   {len(val_data)} răng")
        print(f"{'='*70}")

        # Tạo dataloaders cho fold hiện tại
        train_loader, val_loader = get_fold_dataloaders(
            train_data=train_data,
            val_data=val_data,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            oversample_canal=not args.no_oversample,
        )

        # Train config riêng cho fold (checkpoint dir + log riêng)
        fold_ckpt_dir = kfold_root / f"fold{fold_num}"
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

        train_config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_workers=args.num_workers,
            oversample_canal=not args.no_oversample,
            checkpoint_dir=str(fold_ckpt_dir),
            experiment_name=f"{args.experiment}/fold{fold_num}",
        )

        trainer = KFoldTrainer(
            data_config=data_config,
            model_config=model_config,
            train_config=train_config,
            aug_config=aug_config,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        trainer.train()

        # Lưu kết quả fold này
        fold_results.append({
            "fold": fold_num,
            "n_train": len(train_data),
            "n_val": len(val_data),
            "train_cases": sorted(set(d["case_id"] for d in train_data)),
            "val_cases": sorted(set(d["case_id"] for d in val_data)),
            "best_val_dice": trainer.best_val_dice,
        })

        # Giải phóng VRAM
        del trainer, train_loader, val_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Tổng hợp kết quả K-fold
    total_time = time.time() - t_start
    summary_path = kfold_root / "kfold_summary.json"

    if fold_results:
        best_dices = [r["best_val_dice"] for r in fold_results]
        summary = {
            "experiment": args.experiment,
            "architecture": args.arch,
            "n_folds": n_folds,
            "epochs_per_fold": args.epochs,
            "total_time_hours": total_time / 3600,
            "mean_dice": float(np.mean(best_dices)),
            "std_dice": float(np.std(best_dices)),
            "min_dice": float(np.min(best_dices)),
            "max_dice": float(np.max(best_dices)),
            "folds": fold_results,
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print(f"K-FOLD CV KẾT QUẢ ({len(fold_results)} fold)")
        print(f"{'='*70}")
        for r in fold_results:
            print(f"  Fold {r['fold']}: dice={r['best_val_dice']:.4f} "
                  f"(val cases={r['val_cases']})")
        print(f"\n  Mean dice: {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
        print(f"  Range:     [{summary['min_dice']:.4f}, {summary['max_dice']:.4f}]")
        print(f"  Tổng thời gian: {total_time/3600:.2f}h")
        print(f"  Summary đã lưu: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="K-fold Cross-Validation cho CBCT Tooth & Canal Segmentation"
    )
    parser.add_argument("--data_dir", type=str, default="./data/teeth")
    parser.add_argument("--arch", type=str, default="nnunet",
                        choices=["nnunet", "unet3d", "swin_unetr"])
    parser.add_argument("--n_folds", type=int, default=4,
                        help="Số fold (mặc định 4 - tương ứng 4 ca CBCT)")
    parser.add_argument("--only_fold", type=int, default=None,
                        help="Chỉ chạy 1 fold cụ thể (1-indexed), "
                             "dùng để resume hoặc debug")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_oversample", action="store_true")
    parser.add_argument("--experiment", type=str, default="kfold_tooth_canal")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    run_kfold(args)


if __name__ == "__main__":
    main()
