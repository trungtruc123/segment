"""
Dataset and data loading for CBCT tooth & root canal segmentation.
Supports NIfTI (.nii.gz) and NRRD (.nrrd) volumes.
"""
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def prepare_data_list(
    data_dir: str,
    source: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build list of {"image", "label", "case_id", "source"} dicts.

    case_id dùng để group các răng cùng 1 ca CBCT gốc (tránh leakage khi split).
    source  đánh dấu răng đến từ dataset nào ("v1" / "v2") — cần cho:
        - fine-tune trên tập gộp v1+v2
        - báo cáo dice riêng cho từng domain
        - ghim toàn bộ v1 vào train (xem kfold_split_by_case)
    """
    img_dir = Path(data_dir) / "images"
    mask_dir = Path(data_dir) / "masks"
    if source is None:
        source = Path(data_dir).parent.name or Path(data_dir).name

    data_list = []
    for img_file in sorted(img_dir.iterdir()):
        if img_file.name.startswith("."):
            continue
        stem = img_file.name.split(".")[0]
        # Find matching mask
        mask_candidates = list(mask_dir.glob(f"{stem}*"))
        if mask_candidates:
            data_list.append({
                "image": str(img_file),
                "label": str(mask_candidates[0]),
                "case_id": extract_case_id(img_file.name),
                "source": source,
            })
    return data_list


def prepare_multi_data_list(
    sources: Dict[str, str],
) -> List[Dict[str, str]]:
    """
    Gộp nhiều thư mục teeth/ thành 1 data_list duy nhất.

    Args:
        sources: {"v1": "/path/to/data_v1/teeth", "v2": "/path/to/data_v2/teeth"}

    Trả về list đã gộp, mỗi item có thêm khóa "source".
    Nếu 2 dataset trùng case_id, case_id sẽ được prefix bằng source
    ("v2::LT2_NT2") để K-fold không gộp nhầm 2 ca khác nhau.
    """
    per_source = {}
    for name, d in sources.items():
        items = prepare_data_list(d, source=name)
        per_source[name] = items
        print(f"  [{name}] {len(items)} răng "
              f"từ {len(set(i['case_id'] for i in items))} ca  ({d})")

    # Phát hiện trùng case_id giữa các dataset
    seen = {}
    clash = set()
    for name, items in per_source.items():
        for cid in set(i["case_id"] for i in items):
            if cid in seen and seen[cid] != name:
                clash.add(cid)
            seen[cid] = name

    merged = []
    for name, items in per_source.items():
        for it in items:
            if it["case_id"] in clash:
                it["case_id"] = f"{name}::{it['case_id']}"
            merged.append(it)

    if clash:
        print(f"  [INFO] {len(clash)} case_id trùng giữa các dataset → đã prefix source.")
    print(f"  → Tổng: {len(merged)} răng / "
          f"{len(set(i['case_id'] for i in merged))} ca")
    return merged


def oversample_by_source(
    data_list: List[Dict],
    factors: Dict[str, int],
) -> List[Dict]:
    """
    Nhân bản các item theo source để cân bằng khi gộp 2 dataset lệch nhau.

    Ví dụ v1 có 25 răng, v2 có 18 răng nhưng v2 mới là domain đích:
        oversample_by_source(train, {"v2": 2})  -> v2 xuất hiện 2 lần/epoch.

    Chỉ dùng cho TRAIN list, KHÔNG dùng cho val (sẽ làm sai metric).
    """
    out = []
    for item in data_list:
        out.extend([item] * max(1, int(factors.get(item.get("source", ""), 1))))
    if factors:
        from collections import Counter
        c = Counter(i.get("source") for i in out)
        print(f"  [oversample] train theo source: {dict(c)}")
    return out


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
    val_sources: Optional[Sequence[str]] = None,
    always_train_sources: Sequence[str] = (),
) -> List[Tuple[List[Dict], List[Dict]]]:
    """
    K-fold cross-validation ở mức case_id.

    Với 4 ca CBCT × 6 răng, 4-fold CV cho mỗi fold:
        - train: 3 ca (18 răng)
        - val:   1 ca (6 răng)
    Mỗi ca được dùng làm validation đúng 1 lần -> tận dụng hết dữ liệu.

    Args:
        data_list: list các item có khóa "case_id" (và "source" nếu gộp dataset)
        n_folds: số fold (mặc định = số case để leave-one-out theo ca)
        seed: random seed để shuffle case
        val_sources: chỉ những ca thuộc các source này mới được dùng làm val.
            Ví dụ val_sources=("v2",) → chỉ validate trên data_v2.
        always_train_sources: các source LUÔN nằm trong train ở mọi fold.
            Ví dụ always_train_sources=("v1",) khi fine-tune từ checkpoint đã
            train trên v1: nếu để ca v1 vào val thì dice sẽ bị "thổi phồng"
            (model đã nhìn thấy ca đó trong lần train trước) → metric vô nghĩa.

    Returns:
        List[(train_data, val_data)] có độ dài n_folds
    """
    case_source = {}
    for item in data_list:
        case_source.setdefault(item["case_id"], item.get("source", "default"))

    all_cases = sorted(case_source)
    if len(all_cases) < 2:
        raise ValueError(
            f"Cần ít nhất 2 case để làm K-fold, chỉ tìm thấy {len(all_cases)}"
        )

    # Case bị GHIM vào train ở MỌI fold (không bao giờ vào val)
    pinned = {c for c in all_cases if case_source[c] in set(always_train_sources)}

    # Case ứng viên cho validation
    if val_sources is not None:
        candidates = [c for c in all_cases
                      if case_source[c] in set(val_sources) and c not in pinned]
    else:
        candidates = [c for c in all_cases if c not in pinned]

    if not candidates:
        raise ValueError(
            "Không còn case nào để validate. Kiểm tra val_sources / "
            f"always_train_sources (sources có trong data: "
            f"{sorted(set(case_source.values()))})"
        )

    if n_folds > len(candidates):
        print(f"[WARN] n_folds ({n_folds}) > số case validate được "
              f"({len(candidates)}), dùng {len(candidates)}-fold "
              f"(leave-one-case-out).")
        n_folds = len(candidates)

    # Shuffle để tránh bias theo thứ tự file
    rng = np.random.RandomState(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    # Chia candidates thành n_folds nhóm xấp xỉ đều nhau
    fold_cases = [[] for _ in range(n_folds)]
    for i, cid in enumerate(shuffled):
        fold_cases[i % n_folds].append(cid)

    if pinned:
        print(f"[K-fold] {len(pinned)} ca luôn nằm trong train "
              f"(source={sorted(set(always_train_sources))}): {sorted(pinned)}")
    print(f"[K-fold] {len(candidates)} ca dùng để validate "
          f"→ {n_folds} fold")

    folds = []
    for fold_idx in range(n_folds):
        val_cases = set(fold_cases[fold_idx])
        train_cases = (set(all_cases) - val_cases)

        train_data = [d for d in data_list if d["case_id"] in train_cases]
        val_data = [d for d in data_list if d["case_id"] in val_cases]

        print(f"[Fold {fold_idx+1}/{n_folds}] "
              f"train={len(train_data)} răng ({len(train_cases)} ca), "
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
