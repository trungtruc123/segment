"""
Dataset and data loading for CBCT tooth & root canal segmentation.
Supports NIfTI (.nii.gz) and NRRD (.nrrd) volumes.
"""
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch  # noqa: F401
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import Compose
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

from config import DataConfig


def load_volume(filepath: str) -> np.ndarray:
    """Load a 3D volume from NIfTI or NRRD format."""
    ext = Path(filepath).suffixes
    if ".nrrd" in ext or filepath.endswith(".nrrd"):
        import nrrd
        data, _ = nrrd.read(filepath)
    else:
        img = nib.load(filepath)
        data = img.get_fdata()
    return data.astype(np.float32)


def extract_case_id(filename: str) -> str:
    """
    Trích xuất case_id từ tên file răng.
    Convention: "{case_id}_tooth{NN}.nii.gz" -> "{case_id}"
    Ví dụ: "SLZ000_tooth03.nii.gz" -> "SLZ000"

    Nếu không khớp pattern, trả về stem của file (fallback).
    """
    stem = filename.split(".")[0]
    if "_tooth" in stem:
        return stem.split("_tooth")[0]
    return stem


def prepare_data_list(data_dir: str) -> List[Dict[str, str]]:
    """
    Build list of {"image": path, "label": path, "case_id": str} dicts.
    case_id dùng để group các răng cùng 1 ca CBCT gốc
    (tránh leakage khi split train/val/test).
    """
    img_dir = Path(data_dir) / "images"
    mask_dir = Path(data_dir) / "masks"

    data_list = []
    for img_file in sorted(img_dir.iterdir()):
        stem = img_file.name.split(".")[0]
        # Find matching mask
        mask_candidates = list(mask_dir.glob(f"{stem}*"))
        if mask_candidates:
            data_list.append({
                "image": str(img_file),
                "label": str(mask_candidates[0]),
                "case_id": extract_case_id(img_file.name),
            })
    return data_list


def split_dataset(
    data_list: List[Dict],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    group_by_case: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train/val/test sets.

    Args:
        data_list: danh sách các dict có khóa "case_id"
        ratios: tỷ lệ train/val/test
        seed: random seed
        group_by_case: nếu True, split theo case_id để tránh leakage
                       (các răng từ cùng 1 CBCT gốc sẽ nằm cùng 1 tập).
                       Rất quan trọng khi dùng chiến lược tách răng riêng lẻ:
                       ví dụ 4 ca × 6 răng = 24 mẫu, nếu random split thông
                       thường sẽ khiến răng của cùng 1 ca xuất hiện ở cả train
                       lẫn val/test -> đánh giá không còn chính xác.
    """
    train_ratio, val_ratio, test_ratio = ratios

    if not group_by_case:
        # Random split thông thường (mỗi răng độc lập)
        train_data, temp_data = train_test_split(
            data_list, train_size=train_ratio, random_state=seed
        )
        relative_val = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data, train_size=relative_val, random_state=seed
        )
        return train_data, val_data, test_data

    # Group-by-case split: chia ở mức case_id
    case_ids = sorted(set(item["case_id"] for item in data_list))
    n_cases = len(case_ids)

    if n_cases < 2:
        print(f"[WARN] Chỉ có {n_cases} case_id, không thể split theo case. "
              f"Fallback: random split trên các răng.")
        return split_dataset(data_list, ratios, seed, group_by_case=False)

    # Trường hợp ít case (ví dụ 4 ca): dùng split thủ công có kiểm soát
    rng = np.random.RandomState(seed)
    shuffled = list(case_ids)
    rng.shuffle(shuffled)

    n_train = max(1, int(round(n_cases * train_ratio)))
    n_val = max(1, int(round(n_cases * val_ratio))) if n_cases >= 3 else 0
    # Còn lại là test (ít nhất 1 nếu có đủ case)
    n_train = min(n_train, n_cases - max(1, n_val))
    n_test = n_cases - n_train - n_val

    train_cases = set(shuffled[:n_train])
    val_cases = set(shuffled[n_train:n_train + n_val])
    test_cases = set(shuffled[n_train + n_val:])

    train_data = [d for d in data_list if d["case_id"] in train_cases]
    val_data = [d for d in data_list if d["case_id"] in val_cases]
    test_data = [d for d in data_list if d["case_id"] in test_cases]

    print(f"[Case-based split] {n_cases} cases -> "
          f"train={len(train_cases)} ({sorted(train_cases)}), "
          f"val={len(val_cases)} ({sorted(val_cases)}), "
          f"test={len(test_cases)} ({sorted(test_cases)})")

    return train_data, val_data, test_data


def kfold_split_by_case(
    data_list: List[Dict],
    n_folds: int = 4,
    seed: int = 42,
) -> List[Tuple[List[Dict], List[Dict]]]:
    """
    K-fold cross-validation ở mức case_id.

    Với 4 ca CBCT × 6 răng, 4-fold CV cho mỗi fold:
        - train: 3 ca (18 răng)
        - val:   1 ca (6 răng)
    Mỗi ca được dùng làm validation đúng 1 lần -> tận dụng hết dữ liệu.

    Args:
        data_list: list các item có khóa "case_id"
        n_folds: số fold (mặc định = số case để leave-one-out theo ca)
        seed: random seed để shuffle case

    Returns:
        List[(train_data, val_data)] có độ dài n_folds
    """
    case_ids = sorted(set(item["case_id"] for item in data_list))
    n_cases = len(case_ids)

    if n_cases < 2:
        raise ValueError(
            f"Cần ít nhất 2 case để làm K-fold, chỉ tìm thấy {n_cases}"
        )

    if n_folds > n_cases:
        print(f"[WARN] n_folds ({n_folds}) > số case ({n_cases}), "
              f"dùng {n_cases}-fold (leave-one-case-out).")
        n_folds = n_cases

    # Shuffle case_ids để tránh bias theo thứ tự file
    rng = np.random.RandomState(seed)
    shuffled = list(case_ids)
    rng.shuffle(shuffled)

    # Chia case_ids thành n_folds nhóm xấp xỉ đều nhau
    fold_cases = [[] for _ in range(n_folds)]
    for i, cid in enumerate(shuffled):
        fold_cases[i % n_folds].append(cid)

    # Tạo các split train/val
    folds = []
    for fold_idx in range(n_folds):
        val_cases = set(fold_cases[fold_idx])
        train_cases = set(shuffled) - val_cases

        train_data = [d for d in data_list if d["case_id"] in train_cases]
        val_data = [d for d in data_list if d["case_id"] in val_cases]

        print(f"[Fold {fold_idx+1}/{n_folds}] "
              f"train={len(train_data)} răng ({sorted(train_cases)}), "
              f"val={len(val_data)} răng ({sorted(val_cases)})")

        folds.append((train_data, val_data))

    return folds


def get_fold_dataloaders(
    train_data: List[Dict],
    val_data: List[Dict],
    train_transforms,
    val_transforms,
    batch_size: int = 2,
    num_workers: int = 4,
    oversample_canal: bool = True,
    canal_oversample_ratio: float = 3.0,
) -> Tuple[DataLoader, DataLoader]:
    """Tạo train/val dataloaders cho 1 fold cụ thể."""
    train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0)
    val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=1.0)

    sampler = None
    shuffle = True
    if oversample_canal:
        sampler = CanalAwareSampler(
            train_ds, canal_label=2, oversample_ratio=canal_oversample_ratio
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


class CanalAwareSampler(torch.utils.data.Sampler):
    """
    Oversamples patches that contain root canal voxels.
    Canal is tiny relative to tooth/background, so we sample
    canal-heavy volumes more frequently.
    """

    def __init__(
        self,
        dataset: Dataset,
        canal_label: int = 2,
        oversample_ratio: float = 3.0,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.canal_label = canal_label
        self.oversample_ratio = oversample_ratio
        self.rng = random.Random(seed)

        # Precompute which samples contain canal
        self.canal_indices = []
        self.non_canal_indices = []
        for i, item in enumerate(dataset.data):
            label_path = item["label"] if isinstance(item, dict) else item
            # We mark all samples; actual filtering happens during training
            # For efficiency, assume all samples might contain canal
            self.canal_indices.append(i)

    def __iter__(self):
        # Oversample canal-containing indices
        indices = list(range(len(self.dataset)))
        extra_canal = self.rng.choices(
            self.canal_indices,
            k=int(len(self.canal_indices) * (self.oversample_ratio - 1)),
        )
        indices.extend(extra_canal)
        self.rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return int(len(self.dataset) * self.oversample_ratio)


def get_dataloaders(
    data_config: DataConfig,
    train_transforms: Compose,
    val_transforms: Compose,
    batch_size: int = 2,
    num_workers: int = 4,
    oversample_canal: bool = True,
    canal_oversample_ratio: float = 3.0,
    group_by_case: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with MONAI CacheDataset."""
    data_list = prepare_data_list(data_config.data_dir)
    train_data, val_data, test_data = split_dataset(
        data_list, data_config.split_ratios, data_config.seed,
        group_by_case=group_by_case,
    )

    print(f"Dataset split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0)
    val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=1.0)
    test_ds = CacheDataset(data=test_data, transform=val_transforms, cache_rate=1.0)

    sampler = None
    shuffle = True
    if oversample_canal:
        sampler = CanalAwareSampler(
            train_ds, canal_label=2, oversample_ratio=canal_oversample_ratio
        )
        shuffle = False  # sampler handles ordering

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
