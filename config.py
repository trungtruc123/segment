"""
Configuration for CBCT Tooth & Root Canal Segmentation Pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class DataConfig:
    data_dir: str = "./data"
    # Subdirectories expected:
    #   data/images/   -> .nii.gz or .nrrd CBCT volumes
    #   data/masks/    -> .nii.gz or .nrrd label masks
    # Label mapping: 0=background, 1=tooth, 2=root_canal
    num_classes: int = 3
    class_names: List[str] = field(default_factory=lambda: ["background", "tooth", "root_canal"])
    patch_size: Tuple[int, int, int] = (96, 96, 96)
    spacing: Tuple[float, float, float] = (0.3, 0.3, 0.3)  # target voxel spacing in mm
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)  # train/val/test
    seed: int = 42
    # Group-by-case split: các răng tách ra từ cùng 1 file CBCT gốc
    # sẽ nằm cùng 1 tập (train/val/test) để tránh data leakage.
    # Convention tên file: "{case_id}_tooth{NN}.nii.gz"
    group_by_case: bool = True


@dataclass
class AugmentConfig:
    rotation_range: float = 15.0  # degrees
    elastic_alpha: float = 100.0
    elastic_sigma: float = 10.0
    contrast_range: Tuple[float, float] = (0.75, 1.25)
    brightness_range: Tuple[float, float] = (-0.1, 0.1)
    horizontal_flip_prob: float = 0.5
    vertical_flip: bool = False  # anatomically unrealistic for dental CBCT


@dataclass
class ModelConfig:
    architecture: str = "nnunet"  # "nnunet" | "unet3d" | "swin_unetr"
    in_channels: int = 1
    num_classes: int = 3
    # UNet3D settings
    unet_features: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    # Swin-UNETR settings
    swin_feature_size: int = 48
    swin_depths: Tuple[int, ...] = (2, 2, 2, 2)
    swin_num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    use_pretrained: bool = True  # use MONAI pretrained weights


@dataclass
class TrainConfig:
    # Hyperparameters đã tune riêng cho dataset nhỏ (24 răng / 4 CBCT)
    # và kiến trúc nnU-Net (DynUNet) trên Colab Pro T4.
    epochs: int = 200
    batch_size: int = 4              # nnU-Net nhẹ, T4 16GB OK với patch 96³
    num_workers: int = 2             # Colab shared memory limit
    learning_rate: float = 3e-4      # nnU-Net AdamW standard
    weight_decay: float = 3e-5       # nnU-Net standard
    scheduler: str = "cosine"        # "cosine" or "poly"
    warmup_epochs: int = 5
    # Loss: Dice + Focal với class weights
    # class_weights áp dụng cho cả dice và focal loss.
    # Canal x5 vì là class nhỏ nhất và quan trọng nhất.
    dice_weight: float = 1.0
    focal_weight: float = 1.0
    focal_gamma: float = 2.0
    class_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 5.0])  # bg, tooth, canal
    # Canal oversampling
    oversample_canal: bool = True
    canal_oversample_ratio: float = 3.0
    # Two-stage pipeline
    two_stage: bool = False
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10
    early_stopping_patience: int = 25
    # Mixed precision
    use_amp: bool = True
    # Logging
    log_dir: str = "./logs"
    experiment_name: str = "cbct_tooth_canal_seg"


@dataclass
class InferenceConfig:
    checkpoint_path: str = "./checkpoints/best_model.pth"
    input_dir: str = "./data/test_images"
    output_dir: str = "./predictions"
    patch_size: Tuple[int, int, int] = (96, 96, 96)
    overlap: float = 0.5  # sliding window overlap
    use_tta: bool = True  # test-time augmentation
