"""
Fine-tune (pretrain tiếp) model đã train trên data_v1 sang tập dữ liệu mới.

Bối cảnh:
    - `train_kfold.py` đã train 5-fold trên data_v1 (25 răng / 5 ca CBCT),
      fold1 cho best dice 0.9605 -> checkpoints_v1/.../fold1/best_model.pth
    - Nay có data_v2 (9 ca CBCT, ~2 răng/ca) với đặc tính KHÁC data_v1:
        * shape 714³ @0.07mm (v1: 503³ @0.08mm, riêng SLZ004 512x512x384 @0.16mm)
        * ảnh gốc lưu float64 (v1: int16)  -> nặng RAM gấp 8 lần
        * thang intensity khác hẳn (v2: [-1456, 6230], v1: [-3140, 12380])
      -> xem README_finetune.md để biết chi tiết ảnh hưởng.

Chiến lược mặc định (đã cân nhắc cho dataset nhỏ + domain shift vừa phải):
    1. Train trên tập GỘP v1 + v2 để tránh catastrophic forgetting.
    2. K-fold CV nhưng CHỈ validate trên ca của v2, còn toàn bộ v1 luôn nằm
       trong train. Lý do: checkpoint xuất phát đã "nhìn thấy" 4/5 ca v1 trong
       lần train trước, nếu để ca v1 vào val thì dice sẽ bị thổi phồng.
       Muốn tắt: --val_sources all
    3. LR thấp hơn lần train gốc (1e-4 so với 3e-4) + warmup, optimizer khởi
       tạo lại từ đầu (KHÔNG load optimizer state của lần train trước).
    4. Tuỳ chọn freeze encoder N epoch đầu: --freeze_encoder_epochs 20

Cách dùng:
    # Full fine-tune (khuyến nghị)
    python finetune.py \
        --teeth_dir_v1 data_v1/teeth --teeth_dir_v2 data_v2/teeth \
        --pretrained checkpoints_v1/kfold_nnunet_tooth_canal/fold1/best_model.pth \
        --checkpoint_dir checkpoints_v2 --experiment kfold_ft_v2 \
        --n_folds 5 --epochs 100 --lr 1e-4

    # Freeze encoder 20 epoch đầu rồi mở toàn bộ
    python finetune.py ... --freeze_encoder_epochs 20 --head_lr 3e-4 --lr 5e-5

    # Preset A100 40GB
    python finetune.py ... --batch_size 8 --num_workers 6 --amp_dtype bf16 \
        --sw_batch_size 8 --lr 1.4e-4

    # Chỉ in ra cách chia fold, không train (kiểm tra nhanh, không cần GPU)
    python finetune.py ... --dry_run
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from config import AugmentConfig, DataConfig, ModelConfig, TrainConfig
from dataset import (
    get_fold_dataloaders,
    kfold_split_by_case,
    oversample_by_source,
    prepare_multi_data_list,
)
from train import Trainer, compute_metrics
from transforms import get_train_transforms, get_val_transforms

# Tên các submodule của DynUNet (MONAI) thuộc phần encoder.
# Decoder = upsamples + output_block + deep_supervision_heads.
ENCODER_PREFIXES = ("input_block", "downsamples", "bottleneck")


def setup_gpu(verbose: bool = True) -> dict:
    """
    Bật các tối ưu phụ thuộc GPU và trả về thông tin để gợi ý siêu tham số.

    - TF32: Ampere (A100/L4/RTX30+) chạy matmul/conv fp32 trên tensor core với
      độ chính xác TF32 -> nhanh hơn nhiều, chất lượng segmentation không đổi.
      KHÔNG có tác dụng trên T4 (Turing).
    - cudnn.benchmark: cho cuDNN tự benchmark thuật toán conv tốt nhất ở lần
      chạy đầu. Chỉ có lợi khi shape input CỐ ĐỊNH — đúng với pipeline này
      (patch 96³ khi train, roi 96³ khi sliding-window val).
    - bf16: Ampere trở lên hỗ trợ bfloat16 — dynamic range bằng fp32 nên không
      cần GradScaler và không bị inf/overflow như fp16.
    """
    info = {"name": "cpu", "vram_gb": 0.0, "bf16": False, "tf32": False}
    if not torch.cuda.is_available():
        if verbose:
            print("[GPU] Không thấy CUDA — sẽ chạy trên CPU (rất chậm).")
        return info

    props = torch.cuda.get_device_properties(0)
    major = props.major
    info.update(
        name=props.name,
        vram_gb=props.total_memory / 1024 ** 3,
        bf16=major >= 8,          # Ampere trở lên
        tf32=major >= 8,
    )

    torch.backends.cudnn.benchmark = True
    if info["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if verbose:
        print(f"[GPU] {info['name']}  {info['vram_gb']:.1f} GB  "
              f"(compute {major}.{props.minor})")
        print(f"      cudnn.benchmark=True  TF32={info['tf32']}  "
              f"bf16 khả dụng={info['bf16']}")
        if info["vram_gb"] >= 35:
            print("      → Gợi ý A100/H100: --batch_size 8 --num_workers 6 "
                  "--amp_dtype bf16 --sw_batch_size 8")
        elif info["vram_gb"] >= 20:
            print("      → Gợi ý L4/A10 (24GB): --batch_size 6 --num_workers 4 "
                  "--amp_dtype bf16")
        else:
            print("      → Gợi ý T4 (16GB): --batch_size 4 --num_workers 2 "
                  "--amp_dtype fp16")
    return info


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_pretrained_weights(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
) -> dict:
    """
    Nạp weights từ checkpoint của lần train trước vào model mới.

    Chỉ nạp `model_state_dict` — KHÔNG nạp optimizer/scheduler state, vì:
      - dữ liệu mới, LR mới, số epoch mới -> momentum cũ không còn phù hợp
      - optimizer state của cosine schedule cũ sẽ ép LR về gần 0 ngay lập tức
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    n_loaded = len(state) - len(unexpected)
    print(f"[Pretrained] {ckpt_path}")
    print(f"  epoch={ckpt.get('epoch', '?')}  "
          f"best_val_dice={ckpt.get('best_val_dice', float('nan')):.4f}")
    print(f"  Đã nạp {n_loaded}/{len(state)} tensor")
    if missing:
        print(f"  [WARN] {len(missing)} key thiếu trong checkpoint: {missing[:5]}")
    if unexpected:
        print(f"  [WARN] {len(unexpected)} key thừa trong checkpoint: {unexpected[:5]}")
    if missing or unexpected:
        print("  [WARN] Kiến trúc không khớp 100%. Kiểm tra lại --arch / "
              "num_classes / patch_size trước khi train tiếp.")
    return ckpt


def set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> int:
    """Bật/tắt gradient cho phần encoder. Trả về số param bị ảnh hưởng."""
    n = 0
    for name, param in model.named_parameters():
        if name.startswith(ENCODER_PREFIXES):
            param.requires_grad = trainable
            n += param.numel()
    return n


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class FineTuneTrainer(Trainer):
    """
    Trainer cho fine-tune:
      - nhận sẵn train_loader / val_loader (tái dùng split K-fold)
      - nạp pretrained weights
      - hỗ trợ freeze encoder N epoch đầu rồi mở toàn bộ với LR thấp hơn
      - báo cáo dice TÁCH RIÊNG theo source (v1 / v2)
    """

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        train_config: TrainConfig,
        aug_config: AugmentConfig,
        train_loader,
        val_loader,
        pretrained: Optional[str] = None,
        freeze_encoder_epochs: int = 0,
        head_lr: Optional[float] = None,
        amp_dtype: str = "fp16",
        sw_batch_size: int = 4,
    ):
        from losses import CombinedLoss, DeepSupervisionLoss
        from model import build_model
        from torch.amp import GradScaler
        from torch.utils.tensorboard import SummaryWriter

        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config
        self.aug_config = aug_config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = build_model(model_config, img_size=data_config.patch_size)

        # === Nạp weights của lần train trên data_v1 ===
        self.pretrained_info = None
        if pretrained:
            self.pretrained_info = load_pretrained_weights(
                self.model, pretrained, torch.device("cpu")
            )
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
        self.criterion = self.criterion.to(self.device)

        # === Freeze encoder (nếu có) ===
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.head_lr = head_lr if head_lr is not None else train_config.learning_rate
        self.encoder_frozen = False
        if freeze_encoder_epochs > 0:
            n = set_encoder_trainable(self.model, False)
            self.encoder_frozen = True
            print(f"[Freeze] Đóng băng encoder ({n:,} param) trong "
                  f"{freeze_encoder_epochs} epoch đầu, LR head={self.head_lr:.1e}")

        self._build_optimizer(
            lr=self.head_lr if self.encoder_frozen else train_config.learning_rate,
            total_epochs=train_config.epochs,
        )

        # === Mixed precision ===
        # bf16 không cần loss scaling (dynamic range = fp32) → tắt GradScaler.
        # fp16 bắt buộc phải có scaler nếu không gradient nhỏ sẽ underflow về 0.
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        use_scaler = train_config.use_amp and self.amp_dtype is torch.float16
        self.scaler = GradScaler("cuda", enabled=use_scaler)
        self.use_amp = train_config.use_amp
        self.sw_batch_size = sw_batch_size
        print(f"  AMP: {amp_dtype}  (GradScaler={'on' if use_scaler else 'off'})  "
              f"sw_batch_size={sw_batch_size}")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = None

        self.writer = SummaryWriter(log_dir=os.path.join(
            train_config.log_dir, train_config.experiment_name
        ))
        Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.best_val_dice = 0.0
        self.patience_counter = 0

    def _build_optimizer(self, lr: float, total_epochs: int):
        """Tạo lại optimizer + scheduler (dùng khi bắt đầu và khi unfreeze)."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=self.train_config.weight_decay
        )
        if self.train_config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=max(total_epochs, 1), T_mult=1
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
                self.optimizer, total_iters=max(total_epochs, 1), power=0.9
            )
        if self.train_config.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                total_iters=self.train_config.warmup_epochs,
            )
        n_train = sum(p.numel() for p in params)
        n_all = sum(p.numel() for p in self.model.parameters())
        print(f"  Optimizer: AdamW lr={lr:.1e}  "
              f"({n_train:,}/{n_all:,} param được train)")

    def maybe_unfreeze(self, epoch: int):
        """Mở khoá encoder khi hết giai đoạn freeze, đồng thời hạ LR."""
        if self.encoder_frozen and epoch >= self.freeze_encoder_epochs:
            n = set_encoder_trainable(self.model, True)
            self.encoder_frozen = False
            remaining = max(self.train_config.epochs - epoch, 1)
            print(f"\n[Unfreeze] Mở encoder ({n:,} param) tại epoch {epoch+1}, "
                  f"LR toàn mạng = {self.train_config.learning_rate:.1e}, "
                  f"cosine trên {remaining} epoch còn lại")
            self.train_config.warmup_epochs = 0  # đã warm rồi, không warmup lại
            self._build_optimizer(
                lr=self.train_config.learning_rate, total_epochs=remaining
            )

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Như Trainer.validate nhưng tách dice theo source (v1 / v2)."""
        from monai.inferers import sliding_window_inference
        from losses import DeepSupervisionLoss

        self.model.eval()
        val_loss = 0.0
        rows = []

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            output = sliding_window_inference(
                images,
                roi_size=self.data_config.patch_size,
                sw_batch_size=self.sw_batch_size,
                predictor=self.model,
                overlap=0.5,
            )

            if isinstance(self.criterion, DeepSupervisionLoss):
                loss = self.criterion.base_loss(output, labels)
            else:
                loss = self.criterion(output, labels)
            val_loss += loss.item()

            preds = output.argmax(dim=1)
            if labels.dim() == 5:
                labels = labels.squeeze(1)
            m = compute_metrics(preds, labels, self.data_config.num_classes)

            src = batch.get("source", ["unknown"])
            src = src[0] if isinstance(src, (list, tuple)) else str(src)
            rows.append((src, m["dice_tooth"], m["dice_canal"]))

        n = max(len(self.val_loader), 1)
        tooth = [r[1] for r in rows]
        canal = [r[2] for r in rows]
        out = {
            "val_loss": val_loss / n,
            "val_dice_tooth": float(np.mean(tooth)) if tooth else 0.0,
            "val_dice_canal": float(np.mean(canal)) if canal else 0.0,
        }
        out["val_dice_mean"] = (out["val_dice_tooth"] + out["val_dice_canal"]) / 2

        # Tách theo source
        for s in sorted(set(r[0] for r in rows)):
            t = [r[1] for r in rows if r[0] == s]
            c = [r[2] for r in rows if r[0] == s]
            out[f"dice_tooth_{s}"] = float(np.mean(t))
            out[f"dice_canal_{s}"] = float(np.mean(c))
            out[f"dice_mean_{s}"] = (out[f"dice_tooth_{s}"]
                                     + out[f"dice_canal_{s}"]) / 2
        return out


# ---------------------------------------------------------------------------
# Vòng lặp train 1 fold (có auto-resume + checkpoint theo giờ)
# ---------------------------------------------------------------------------
def train_one_fold(
    fold_num: int,
    n_folds: int,
    train_data: List[Dict],
    val_data: List[Dict],
    *,
    data_config: DataConfig,
    model_config: ModelConfig,
    aug_config: AugmentConfig,
    train_tfm,
    val_tfm,
    kfold_root: Path,
    log_dir: Path,
    args,
) -> float:
    fold_dir = kfold_root / f"fold{fold_num}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    latest_path = fold_dir / "latest.pth"
    done_path = fold_dir / "DONE"

    if done_path.exists():
        print(f"\n[Fold {fold_num}] Đã xong trước đó, skip. "
              f"(xoá file DONE nếu muốn train lại)")
        ckpt = torch.load(fold_dir / "best_model.pth",
                          map_location="cpu", weights_only=False)
        return float(ckpt.get("best_val_dice", 0.0))

    print(f"\n{'='*70}\nFOLD {fold_num}/{n_folds}\n{'='*70}")
    print(f"  Train: {len(train_data)} răng "
          f"({sorted({d['case_id'] for d in train_data})})")
    print(f"  Val:   {len(val_data)} răng "
          f"({sorted({d['case_id'] for d in val_data})})")

    # Nhân bản v2 trong train nếu muốn ưu tiên domain mới
    train_data_ep = oversample_by_source(
        train_data, {"v2": args.v2_repeat} if args.v2_repeat > 1 else {}
    )

    train_loader, val_loader = get_fold_dataloaders(
        train_data=train_data_ep, val_data=val_data,
        train_transforms=train_tfm, val_transforms=val_tfm,
        batch_size=args.batch_size, num_workers=args.num_workers,
        oversample_canal=not args.no_oversample,
        canal_oversample_ratio=args.canal_oversample_ratio,
    )

    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        oversample_canal=not args.no_oversample,
        canal_oversample_ratio=args.canal_oversample_ratio,
        class_weights=args.class_weights,
        early_stopping_patience=args.patience,
        checkpoint_dir=str(fold_dir),
        experiment_name=f"{args.experiment}/fold{fold_num}",
        log_dir=str(log_dir),
    )

    trainer = FineTuneTrainer(
        data_config=data_config, model_config=model_config,
        train_config=train_config, aug_config=aug_config,
        train_loader=train_loader, val_loader=val_loader,
        pretrained=args.pretrained,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        head_lr=args.head_lr,
        amp_dtype=args.amp_dtype,
        sw_batch_size=args.sw_batch_size,
    )

    # === Auto-resume trong CHÍNH lần fine-tune này ===
    start_epoch = 0
    if latest_path.exists():
        print(f"[Resume] Loading {latest_path}")
        ckpt = torch.load(latest_path, map_location=trainer.device,
                          weights_only=False)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        # nếu đã qua giai đoạn freeze thì mở encoder + dựng lại optimizer
        trainer.maybe_unfreeze(start_epoch)
        if "optimizer_state_dict" in ckpt:
            try:
                trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except ValueError:
                print("  [WARN] optimizer state không khớp (do đổi freeze "
                      "state) → dùng optimizer mới.")
        if "scaler_state_dict" in ckpt:
            trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])
        trainer.best_val_dice = ckpt.get("best_val_dice", 0.0)
        print(f"[Resume] Tiếp tục từ epoch {start_epoch}, "
              f"best_dice={trainer.best_val_dice:.4f}")

    last_save = time.time()
    interval_sec = args.ckpt_interval_h * 3600
    epoch = start_epoch

    for epoch in range(start_epoch, args.epochs):
        trainer.maybe_unfreeze(epoch)

        t0 = time.time()
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate(epoch)
        elapsed = time.time() - t0
        lr_now = trainer.optimizer.param_groups[0]["lr"]

        for k, v in {**train_metrics, **val_metrics}.items():
            trainer.writer.add_scalar(k, v, epoch)
        trainer.writer.add_scalar("LR", lr_now, epoch)

        mean_dice = val_metrics["val_dice_mean"]
        is_best = mean_dice > trainer.best_val_dice
        if is_best:
            trainer.best_val_dice = mean_dice
            trainer.patience_counter = 0
        else:
            trainer.patience_counter += 1

        per_src = "  ".join(
            f"{k.replace('dice_mean_', '')}={v:.4f}"
            for k, v in val_metrics.items() if k.startswith("dice_mean_")
        )
        print(f"Ep [{epoch+1}/{args.epochs}] ({elapsed:.0f}s) lr={lr_now:.2e} | "
              f"train={train_metrics['train_loss']:.4f} "
              f"val={val_metrics['val_loss']:.4f} | "
              f"tooth={val_metrics['val_dice_tooth']:.4f} "
              f"canal={val_metrics['val_dice_canal']:.4f}"
              + (f" | {per_src}" if per_src else "")
              + ("  *BEST*" if is_best else ""))

        if is_best:
            torch.save({
                "epoch": epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": trainer.scheduler.state_dict(),
                "best_val_dice": float(trainer.best_val_dice),
                "metrics": {k: float(v) for k, v in
                            {**train_metrics, **val_metrics}.items()},
                "pretrained_from": args.pretrained,
            }, fold_dir / "best_model.pth")

        if time.time() - last_save >= interval_sec:
            torch.save({
                "epoch": epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": trainer.scheduler.state_dict(),
                "scaler_state_dict": trainer.scaler.state_dict(),
                "best_val_dice": float(trainer.best_val_dice),
            }, latest_path)
            last_save = time.time()
            print(f"  [Checkpoint {args.ckpt_interval_h}h] đã lưu "
                  f"{latest_path.name} tại epoch {epoch+1}")

        if trainer.patience_counter >= args.patience:
            print(f"\nEarly stopping tại epoch {epoch+1} "
                  f"(không cải thiện {args.patience} epoch)")
            break

    torch.save({
        "epoch": epoch,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "scaler_state_dict": trainer.scaler.state_dict(),
        "best_val_dice": float(trainer.best_val_dice),
    }, latest_path)
    done_path.write_text(f"best_val_dice={trainer.best_val_dice:.4f}\n")

    best = float(trainer.best_val_dice)
    trainer.writer.close()
    del trainer, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


# ---------------------------------------------------------------------------
# Điều phối K-fold
# ---------------------------------------------------------------------------
def build_folds(args):
    """Gộp v1 + v2, chia K-fold theo case. Trả về (data_list, folds)."""
    sources = {}
    if args.teeth_dir_v1:
        sources["v1"] = args.teeth_dir_v1
    if args.teeth_dir_v2:
        sources["v2"] = args.teeth_dir_v2
    if not sources:
        raise ValueError("Cần ít nhất --teeth_dir_v1 hoặc --teeth_dir_v2")

    print("\n=== Dữ liệu ===")
    data_list = prepare_multi_data_list(sources)

    if args.val_sources == "all":
        val_sources, always_train = None, ()
        print("\n[Split] validate trên MỌI source. Lưu ý: dice trên ca v1 sẽ "
              "lạc quan hơn thực tế vì checkpoint xuất phát đã học các ca đó.")
    else:
        val_sources = tuple(args.val_sources.split(","))
        always_train = tuple(s for s in sources if s not in val_sources)

    folds = kfold_split_by_case(
        data_list,
        n_folds=args.n_folds,
        seed=args.seed,
        val_sources=val_sources,
        always_train_sources=always_train,
    )
    return data_list, folds


def run_finetune(args):
    gpu = setup_gpu()
    if args.amp_dtype == "bf16" and not gpu["bf16"]:
        print("[WARN] GPU không hỗ trợ bf16 → tự chuyển về fp16.")
        args.amp_dtype = "fp16"

    data_config = DataConfig(
        data_dir=args.teeth_dir_v2 or args.teeth_dir_v1,
        patch_size=tuple(args.patch_size),
    )
    model_config = ModelConfig(architecture=args.arch)
    aug_config = AugmentConfig()

    data_list, folds = build_folds(args)
    n_folds = len(folds)

    if args.dry_run:
        print("\n[dry_run] Chỉ kiểm tra split, không train. Thoát.")
        return

    train_tfm = get_train_transforms(data_config, aug_config)
    val_tfm = get_val_transforms(data_config)

    kfold_root = Path(args.checkpoint_dir) / args.experiment
    kfold_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)

    print(f"\n=== Siêu tham số fine-tune ===")
    print(f"  Pretrained:     {args.pretrained}")
    print(f"  Architecture:   {args.arch}")
    print(f"  Epochs:         {args.epochs}   (gốc: 200)")
    print(f"  LR:             {args.lr}   (gốc: 3e-4)")
    print(f"  Freeze encoder: {args.freeze_encoder_epochs} epoch "
          f"(head_lr={args.head_lr})")
    print(f"  Batch size:     {args.batch_size} "
          f"(= {args.batch_size * 2} patch/step, do num_samples=2)")
    print(f"  Patch size:     {tuple(args.patch_size)}")
    print(f"  AMP dtype:      {args.amp_dtype}")
    print(f"  num_workers:    {args.num_workers}")
    print(f"  Class weights:  {args.class_weights}")
    print(f"  v2_repeat:      {args.v2_repeat}")
    print(f"  Checkpoint dir: {kfold_root}")

    results = []
    t_start = time.time()
    for fold_idx, (train_data, val_data) in enumerate(folds):
        fold_num = fold_idx + 1
        if args.only_fold is not None and fold_num != args.only_fold:
            continue
        train_one_fold(
            fold_num, n_folds, train_data, val_data,
            data_config=data_config, model_config=model_config,
            aug_config=aug_config, train_tfm=train_tfm, val_tfm=val_tfm,
            kfold_root=kfold_root, log_dir=log_dir, args=args,
        )
        best_ckpt = kfold_root / f"fold{fold_num}" / "best_model.pth"
        if best_ckpt.exists():
            ck = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            m = ck.get("metrics", {})
            results.append({
                "fold": fold_num,
                "best_val_dice": float(ck.get("best_val_dice", 0.0)),
                "dice_tooth": m.get("val_dice_tooth"),
                "dice_canal": m.get("val_dice_canal"),
                "val_cases": sorted({d["case_id"] for d in val_data}),
                "val_sources": sorted({d.get("source") for d in val_data}),
                "n_train": len(train_data),
                "n_val": len(val_data),
            })

    if results:
        dices = [r["best_val_dice"] for r in results]
        summary = {
            "experiment": args.experiment,
            "mode": "finetune",
            "pretrained_from": args.pretrained,
            "architecture": args.arch,
            "n_folds": len(results),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "freeze_encoder_epochs": args.freeze_encoder_epochs,
            "class_weights": args.class_weights,
            "val_sources": args.val_sources,
            "v2_repeat": args.v2_repeat,
            "total_time_hours": (time.time() - t_start) / 3600,
            "mean_dice": float(np.mean(dices)),
            "std_dice": float(np.std(dices)),
            "folds": results,
        }
        out = kfold_root / "kfold_summary.json"
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        print(f"\n{'='*70}\nFINE-TUNE HOÀN TẤT\n{'='*70}")
        for r in results:
            print(f"  Fold {r['fold']}: dice={r['best_val_dice']:.4f} "
                  f"(val={r['val_cases']})")
        print(f"\n  Mean dice: {summary['mean_dice']:.4f} "
              f"± {summary['std_dice']:.4f}")
        print(f"  Summary: {out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fine-tune CBCT tooth/canal segmentation từ checkpoint có sẵn"
    )
    # Dữ liệu
    p.add_argument("--teeth_dir_v1", type=str, default="data_v1/teeth",
                   help="Thư mục teeth/ đã split của data_v1 (để rỗng nếu chỉ train v2)")
    p.add_argument("--teeth_dir_v2", type=str, default="data_v2/teeth",
                   help="Thư mục teeth/ đã split của data_v2")
    p.add_argument("--val_sources", type=str, default="v2",
                   help="Source được dùng làm validation ('v2', 'v1,v2', hoặc 'all')")
    p.add_argument("--v2_repeat", type=int, default=1,
                   help="Nhân bản mỗi răng v2 trong train N lần (cân bằng domain)")
    # Checkpoint
    p.add_argument("--pretrained", type=str,
                   default="checkpoints_v1/kfold_nnunet_tooth_canal/fold1/best_model.pth")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints_v2")
    p.add_argument("--experiment", type=str, default="kfold_ft_v2")
    p.add_argument("--log_dir", type=str, default="./logs")
    # Model
    p.add_argument("--arch", type=str, default="nnunet",
                   choices=["nnunet", "unet3d", "swin_unetr"])
    p.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    # Fine-tune
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="LR khi train toàn mạng (thấp hơn 3e-4 của lần train gốc)")
    p.add_argument("--head_lr", type=float, default=3e-4,
                   help="LR cho giai đoạn freeze encoder")
    p.add_argument("--freeze_encoder_epochs", type=int, default=0,
                   help="0 = full fine-tune ngay từ đầu (khuyến nghị). "
                        ">0 = chỉ train decoder+head trong N epoch đầu.")
    p.add_argument("--weight_decay", type=float, default=3e-5)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Số ITEM mỗi step; mỗi item cho 2 patch (num_samples=2) "
                        "→ patch/step = batch_size × 2. T4: 4 | L4: 6 | A100: 8")
    p.add_argument("--num_workers", type=int, default=2,
                   help="Colab T4 nên để 2 (shared memory nhỏ); A100 để 6")
    p.add_argument("--amp_dtype", type=str, default="fp16",
                   choices=["fp16", "bf16"],
                   help="bf16 chỉ chạy trên Ampere+ (A100/L4/H100), ổn định hơn fp16")
    p.add_argument("--sw_batch_size", type=int, default=4,
                   help="Số patch mỗi lượt sliding-window khi validate. "
                        "A100 để 8 → validate nhanh gần gấp đôi")
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--class_weights", type=float, nargs=3, default=[1.0, 1.0, 5.0])
    p.add_argument("--no_oversample", action="store_true")
    p.add_argument("--canal_oversample_ratio", type=float, default=3.0)
    # K-fold
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--only_fold", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt_interval_h", type=float, default=12.0)
    p.add_argument("--dry_run", action="store_true",
                   help="Chỉ in cách chia fold rồi thoát (không cần GPU)")
    return p.parse_args(argv)


if __name__ == "__main__":
    run_finetune(parse_args())
