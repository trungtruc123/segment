"""
Data augmentation và preprocessing cho CBCT tooth & canal segmentation.

Thiết kế riêng cho dữ liệu đã split thành răng riêng lẻ (volume nhỏ).
Không flip theo trục S-I (crown/root) vì anatomically unrealistic.
"""
from typing import Tuple

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    SpatialPadd,
)

from config import AugmentConfig, DataConfig

# CBCT KHÔNG được calibrated HU như CT - các máy CBCT khác nhau
# output intensity range khác nhau (một số máy dùng raw, một số scaled).
# → Dùng percentile clipping [0.5%, 99.5%] là robust nhất.
PCT_LOW = 0.5
PCT_HIGH = 99.5


def get_train_transforms(
    data_config: DataConfig,
    aug_config: AugmentConfig,
) -> Compose:
    """
    Training transforms với augmentation được tune riêng cho CBCT răng.

    Pipeline:
        Load → Orient → Resample → HU clip → Foreground crop
        → Random patch crop (canal-aware)
        → Spatial aug (affine + elastic + flip)
        → Intensity aug (noise + smooth + bias + gamma + scale/shift)
    """
    keys = ["image", "label"]
    return Compose([
        # ==== Load và chuẩn hóa hình học ====
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),  # thay AddChanneld (deprecated)
        Orientationd(keys=keys, axcodes="RAS"),

        # Resample về spacing đồng nhất (isotropic 0.3mm)
        Spacingd(
            keys=keys,
            pixdim=data_config.spacing,
            mode=("bilinear", "nearest"),
        ),

        # ==== Intensity normalization ====
        # Percentile clipping [0.5, 99.5] → scale về [0, 1]
        # Robust với CBCT vì không phụ thuộc vào HU calibration
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=PCT_LOW, upper=PCT_HIGH,
            b_min=0.0, b_max=1.0,
            clip=True,
            relative=False,
        ),

        # Crop foreground với margin rộng để giữ đầy đủ chân răng (apex)
        CropForegroundd(keys=keys, source_key="image", margin=20),

        # ==== Random patch extraction (canal-aware) ====
        # Với răng đã split (volume nhỏ), num_samples=2 đủ tránh overlap
        # pos=4 để tăng xác suất sample vào region có canal
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=data_config.patch_size,
            pos=4,
            neg=1,
            num_samples=2,
            allow_smaller=True,
        ),
        SpatialPadd(keys=keys, spatial_size=data_config.patch_size),

        # ==== Spatial augmentation ====
        # Affine: rotate chủ yếu quanh trục S-I (axial), hạn chế 2 trục còn lại
        # để không làm răng nghiêng quá mức
        RandAffined(
            keys=keys,
            prob=0.5,
            rotate_range=[0.087, 0.087, 0.26],  # (±5°, ±5°, ±15°)
            scale_range=[0.1, 0.1, 0.1],
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),

        # Elastic deformation — giảm magnitude để không phá canal (chỉ 1-2mm)
        # Với sigma 5-8 và magnitude 20-60, deformation mượt nhưng không làm
        # mất connectivity của canal
        Rand3DElasticd(
            keys=keys,
            prob=0.3,
            sigma_range=(5, 8),
            magnitude_range=(20, 60),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),

        # Chỉ flip trục R-L (axis 0). KHÔNG flip trục S-I (axis 2 = crown/root).
        RandFlipd(keys=keys, prob=aug_config.horizontal_flip_prob, spatial_axis=0),

        # ==== Intensity augmentation ====
        # Bias field: mô phỏng intensity inhomogeneity do artifact CBCT
        RandBiasFieldd(
            keys=["image"],
            prob=0.3,
            coeff_range=(0.0, 0.1),
            degree=3,
        ),

        # Gaussian noise (thermal noise của detector)
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.02),

        # Gaussian smooth (mô phỏng defocus / PSF khác nhau)
        RandGaussianSmoothd(
            keys=["image"],
            prob=0.15,
            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0),
        ),

        # Gamma correction — quan trọng cho CBCT vì contrast window
        # giữa các máy rất khác nhau
        RandAdjustContrastd(
            keys=["image"],
            prob=0.3,
            gamma=(0.7, 1.5),
        ),

        # Scale/shift intensity (sau [0,1] scale, offsets nhỏ hơn)
        RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.3),

        EnsureTyped(keys=keys),
    ])


def get_val_transforms(data_config: DataConfig) -> Compose:
    """Validation/test transforms — preprocessing giống train nhưng không aug."""
    keys = ["image", "label"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=data_config.spacing,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=PCT_LOW, upper=PCT_HIGH,
            b_min=0.0, b_max=1.0,
            clip=True,
            relative=False,
        ),
        CropForegroundd(keys=keys, source_key="image", margin=20),
        EnsureTyped(keys=keys),
    ])
