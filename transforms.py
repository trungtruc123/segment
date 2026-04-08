"""
Data augmentation and preprocessing transforms using MONAI.
No vertical flip (anatomically unrealistic for dental CBCT).
"""
from typing import Tuple

from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

from config import AugmentConfig, DataConfig


def get_train_transforms(
    data_config: DataConfig,
    aug_config: AugmentConfig,
) -> Compose:
    """Training transforms with augmentation."""
    keys = ["image", "label"]
    return Compose([
        # Load and orient
        LoadImaged(keys=keys),
        AddChanneld(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),

        # Resample to target spacing
        Spacingd(
            keys=keys,
            pixdim=data_config.spacing,
            mode=("bilinear", "nearest"),
        ),

        # Intensity normalization (z-score)
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # Crop foreground to reduce empty space
        CropForegroundd(keys=keys, source_key="image", margin=10),

        # Random patch extraction — oversample positive (tooth/canal) regions
        # pos=2 means 2x more likely to sample from labeled regions
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=data_config.patch_size,
            pos=3,  # heavily favor patches with labels (canal awareness)
            neg=1,
            num_samples=4,  # patches per volume
        ),

        # Pad if volume is smaller than patch size
        SpatialPadd(keys=keys, spatial_size=data_config.patch_size),

        # Spatial augmentation: rotation + scaling (no elastic in MONAI by default)
        RandAffined(
            keys=keys,
            prob=0.3,
            rotate_range=[0.26, 0.26, 0.26],  # ~15 degrees
            scale_range=[0.1, 0.1, 0.1],
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),

        # Horizontal flip only (no vertical — anatomically unrealistic)
        RandFlipd(keys=keys, prob=aug_config.horizontal_flip_prob, spatial_axis=0),

        # Intensity augmentation
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
        RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.5, 1.0)),
        RandScaleIntensityd(keys=["image"], factors=0.25, prob=0.2),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),

        # Convert to tensor
        EnsureTyped(keys=keys),
    ])


def get_val_transforms(data_config: DataConfig) -> Compose:
    """Validation/test transforms — no augmentation."""
    keys = ["image", "label"]
    return Compose([
        LoadImaged(keys=keys),
        AddChanneld(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=data_config.spacing,
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=keys, source_key="image", margin=10),
        EnsureTyped(keys=keys),
    ])
