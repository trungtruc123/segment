"""
Full-CBCT inference pipeline (không cần label).

Pipeline:
    1. Load CBCT volume (N răng, ví dụ N=6)
    2. Auto-detect từng răng bằng intensity threshold + connected component
       (find_teeth_from_image từ split_teeth.py)
    3. Với mỗi răng phát hiện được:
        a) Crop bounding box có margin
        b) Tiền xử lý bằng MONAI val transforms (Orientation RAS, Spacing,
           ScaleIntensityRangePercentiles)
        c) Chạy sliding_window_inference
        d) Resample prediction về lại kích thước crop gốc (invert spacing)
        e) Paste vào volume output full-size theo bbox
    4. Hợp nhất: ưu tiên canal > tooth > background khi có overlap
    5. Lưu 3 file NIfTI:
        - input.nii.gz                  (= CBCT gốc, giữ nguyên)
        - segments_labels_pred.nii.gz   (label gộp 0/1/2, cùng shape + affine)
        - overlay_preview.nii.gz        (tùy chọn)

Cách dùng:
    python inference_full_cbct.py \
        --input data/raw_test/case05.nii.gz \
        --checkpoint checkpoints/kfold_nnunet_tooth_canal/fold1/best_model.pth \
        --output_dir predictions/case05 \
        --arch nnunet

    # Hoặc auto-pick fold tốt nhất từ kfold_summary.json:
    python inference_full_cbct.py \
        --input data/raw_test/case05.nii.gz \
        --kfold_dir checkpoints/kfold_nnunet_tooth_canal \
        --output_dir predictions/case05
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

from scipy import ndimage as ndi

from config import DataConfig, ModelConfig
from model import build_model
from split_teeth import find_teeth_from_image, get_bounding_box


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_best_fold_checkpoint(kfold_dir: Path) -> Path:
    """Đọc kfold_summary.json và trả về path best_model.pth của fold dice cao nhất."""
    summary_file = kfold_dir / "kfold_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Không có {summary_file}")
    with open(summary_file) as f:
        summary = json.load(f)
    best = max(summary["folds"], key=lambda x: x["best_val_dice"])
    ckpt = kfold_dir / f"fold{best['fold']}" / "best_model.pth"
    print(f"Best fold: {best['fold']}  val_dice={best['best_val_dice']:.4f}")
    return ckpt


def preprocess_crop(
    crop: np.ndarray,
    target_spacing: Tuple[float, float, float],
    orig_spacing: Tuple[float, float, float],
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Tiền xử lý crop volume để đưa vào model:
        - Percentile normalize về [0, 1]
        - Resample sang target_spacing (giống val_tfm)
    Trả về tensor (1, 1, D, H, W) và shape gốc trước resample để invert sau.
    """
    orig_shape = crop.shape  # (D, H, W) voxel space

    # Percentile normalize
    lo = np.percentile(crop, percentile_low)
    hi = np.percentile(crop, percentile_high)
    crop_n = np.clip(crop, lo, hi)
    crop_n = (crop_n - lo) / (hi - lo + 1e-8)

    # Tensor (1, 1, D, H, W)
    x = torch.from_numpy(crop_n.astype(np.float32))[None, None].to(device)

    # Resample: scale = orig_spacing / target_spacing
    scale = [float(o) / float(t) for o, t in zip(orig_spacing, target_spacing)]
    new_shape = tuple(max(1, int(round(s * sc))) for s, sc in zip(orig_shape, scale))
    if new_shape != orig_shape:
        x = F.interpolate(x, size=new_shape, mode="trilinear", align_corners=False)
    return x, orig_shape


def postprocess_prediction(
    logits: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Resample logits về target_shape rồi argmax → numpy uint8.

    logits: (1, C, D, H, W) — nếu nnU-Net deep supervision có thể là 6D,
            đã được unpack trước khi gọi hàm này.
    """
    if tuple(logits.shape[2:]) != target_shape:
        logits = F.interpolate(
            logits, size=target_shape, mode="trilinear", align_corners=False
        )
    prob = torch.softmax(logits, dim=1)
    pred = prob.argmax(dim=1).squeeze(0).to(torch.uint8).cpu().numpy()
    return pred


def merge_prediction_into_full(
    full_label: np.ndarray,
    tooth_pred: np.ndarray,
    bbox: Tuple[slice, slice, slice],
):
    """
    Paste tooth_pred vào vị trí bbox của full_label, theo luật ưu tiên:
        canal (2) > tooth (1) > background (0)
    Không ghi đè voxel đã được gán canal bởi răng khác.
    """
    region = full_label[bbox]
    # Canal: luôn ghi đè (trừ khi voxel hiện tại đã là canal)
    canal_mask = tooth_pred == 2
    region[canal_mask] = 2
    # Tooth: chỉ ghi nếu voxel hiện tại chưa phải canal
    tooth_mask = (tooth_pred == 1) & (region != 2)
    region[tooth_mask] = 1
    full_label[bbox] = region


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load checkpoint ---
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.kfold_dir:
        ckpt_path = find_best_fold_checkpoint(Path(args.kfold_dir))
    else:
        raise ValueError("Cần --checkpoint hoặc --kfold_dir")

    model_cfg = ModelConfig(architecture=args.arch)
    model = build_model(model_cfg, img_size=tuple(args.patch_size)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

    # --- Load input CBCT ---
    input_path = Path(args.input)
    print(f"\nLoading CBCT: {input_path}")
    nii = nib.load(str(input_path))
    image = nii.get_fdata()
    affine = nii.affine
    header = nii.header
    # Spacing từ affine (lấy độ dài cột đầu)
    orig_spacing = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
    print(f"  Shape: {image.shape}")
    print(f"  Spacing: {orig_spacing}  (mm)")

    # --- Detect teeth ---
    # Nếu spacing nhỏ (<0.2mm) → downsample về ~0.5mm để detect nhanh + chính xác hơn
    # (ở resolution cao, morphological closing không đủ phủ khoang tủy,
    #  và toàn bộ răng dễ merge thành 1 component bị xóa nhầm)
    DETECT_SPACING = 0.25  # mm — downsample cho detect nhanh nhưng giữ biên
    min_spacing = min(orig_spacing)
    need_downsample = min_spacing < 0.2

    print(f"\n[1/3] Đang detect từng răng từ image...")
    t0 = time.time()

    if need_downsample:
        downsample_factor = tuple(
            DETECT_SPACING / s for s in orig_spacing
        )
        print(f"  Spacing {min_spacing:.4f}mm < 0.2mm → downsample {downsample_factor[0]:.1f}x "
              f"về ~{DETECT_SPACING}mm để detect")
        # Downsample bằng scipy zoom (order=1 = bilinear, nhanh)
        image_ds = ndi.zoom(image, [1.0 / f for f in downsample_factor], order=1)
        print(f"  Downsampled shape: {image_ds.shape}")

        # Min_voxels scale theo tỉ lệ thể tích
        vol_ratio = np.prod(downsample_factor)
        min_voxels_ds = max(100, int(args.min_voxels / vol_ratio))

        components_ds, num_teeth = find_teeth_from_image(
            image_ds,
            percentile_low=args.percentile_threshold,
            min_voxels=min_voxels_ds,
        )

        # Upsample component map về resolution gốc (nearest neighbor giữ integer label)
        components = ndi.zoom(
            components_ds, downsample_factor, order=0  # order=0 = nearest
        )
        # Crop/pad nếu shape bị lệch 1 voxel do rounding
        if components.shape != image.shape:
            out = np.zeros(image.shape, dtype=components.dtype)
            slices = tuple(slice(0, min(c, o)) for c, o in zip(components.shape, image.shape))
            out[slices] = components[slices]
            components = out
    else:
        components, num_teeth = find_teeth_from_image(
            image,
            percentile_low=args.percentile_threshold,
            min_voxels=args.min_voxels,
        )

    t_detect = time.time() - t0
    print(f"  Tìm thấy {num_teeth} răng trong {t_detect:.1f}s")
    if num_teeth == 0:
        raise RuntimeError(
            f"Không detect được răng nào.\n"
            f"  Thử giảm --percentile_threshold (hiện={args.percentile_threshold}, "
            f"thử 50 hoặc 40)\n"
            f"  Hoặc giảm --min_voxels (hiện={args.min_voxels}, thử 1000)"
        )

    # --- Inference từng răng + stitch ---
    print(f"\n[2/3] Inference từng răng + ghép lại...")
    full_label = np.zeros(image.shape, dtype=np.uint8)
    target_spacing = tuple(args.target_spacing)
    patch_size = tuple(args.patch_size)

    # Margin tính theo mm (mặc định ~5mm) rồi convert sang voxels
    margin_mm = args.margin_mm
    margin_voxels = max(5, int(round(margin_mm / min_spacing)))
    print(f"  Margin: {margin_mm}mm = {margin_voxels} voxels (spacing={min_spacing:.4f}mm)")

    timings = []
    tooth_stats = []
    for tooth_idx in range(1, num_teeth + 1):
        this_tooth_mask = components == tooth_idx
        bbox = get_bounding_box(this_tooth_mask, margin=margin_voxels)
        if bbox is None:
            continue

        crop = image[bbox]
        t1 = time.time()

        # Preprocess
        x, orig_crop_shape = preprocess_crop(
            crop,
            target_spacing=target_spacing,
            orig_spacing=orig_spacing,
            device=device,
        )

        # Inference
        with torch.no_grad():
            logits = sliding_window_inference(
                x, roi_size=patch_size, sw_batch_size=2,
                predictor=model, overlap=0.5, mode="gaussian",
            )
            # nnU-Net deep supervision → 6D
            if logits.dim() == 6:
                logits = logits[:, 0]

        # Resample prediction về kích thước crop gốc rồi argmax
        tooth_pred = postprocess_prediction(logits, orig_crop_shape)

        # Paste vào full_label
        merge_prediction_into_full(full_label, tooth_pred, bbox)

        dt = time.time() - t1
        timings.append(dt)
        n_tooth_vx = int((tooth_pred == 1).sum())
        n_canal_vx = int((tooth_pred == 2).sum())
        tooth_stats.append({
            "idx": tooth_idx,
            "bbox_shape": crop.shape,
            "tooth_voxels": n_tooth_vx,
            "canal_voxels": n_canal_vx,
            "time_s": dt,
        })
        print(f"  Răng {tooth_idx:02d}: crop={crop.shape} "
              f"tooth={n_tooth_vx:,}vx canal={n_canal_vx:,}vx "
              f"[{dt*1000:.0f} ms]")

    # --- Save outputs ---
    print(f"\n[3/3] Lưu kết quả...")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # input.nii.gz = CBCT gốc (copy nguyên, giữ affine + header)
    input_out = out_dir / "input.nii.gz"
    nib.save(
        nib.Nifti1Image(image.astype(np.float32), affine, header),
        str(input_out),
    )

    # segments_labels_pred.nii.gz = label gộp full-size
    pred_out = out_dir / "segments_labels_pred.nii.gz"
    pred_nii = nib.Nifti1Image(full_label.astype(np.uint8), affine)
    pred_nii.header.set_data_dtype(np.uint8)
    nib.save(pred_nii, str(pred_out))

    total_tooth = int((full_label == 1).sum())
    total_canal = int((full_label == 2).sum())
    total_time = sum(timings) + t_detect

    print(f"\n{'='*60}")
    print(f"HOÀN TẤT — tổng {total_time:.1f}s")
    print(f"{'='*60}")
    print(f"  Detect răng:         {t_detect:.1f}s")
    print(f"  Inference {num_teeth} răng:    {sum(timings):.1f}s "
          f"(trung bình {np.mean(timings)*1000:.0f} ms/răng)")
    print(f"  Full-CBCT voxels:    tooth={total_tooth:,}  canal={total_canal:,}")
    print(f"\n  📄 {input_out}")
    print(f"  📄 {pred_out}")
    print(f"\nMở 3D Slicer:")
    print(f"  File → Add Data → chọn cả 2 file trên")
    print(f"  File label nhớ tick cột Description = 'Segmentation'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-CBCT inference: tách → infer từng răng → gộp lại"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="File CBCT đầu vào (.nii / .nii.gz)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Thư mục lưu input.nii.gz + segments_labels_pred.nii.gz")

    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path tới best_model.pth (nếu không dùng --kfold_dir)")
    parser.add_argument("--kfold_dir", type=str, default=None,
                        help="Thư mục kfold experiment (auto-pick best fold)")
    parser.add_argument("--arch", type=str, default="nnunet",
                        choices=["nnunet", "unet3d", "swin_unetr"])

    # Inference params
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--target_spacing", type=float, nargs=3,
                        default=[0.3, 0.3, 0.3],
                        help="Spacing model train (phải khớp DataConfig)")

    # Tooth detection
    parser.add_argument("--margin", type=int, default=15,
                        help="(deprecated, dùng --margin_mm) Margin voxel quanh bbox")
    parser.add_argument("--margin_mm", type=float, default=5.0,
                        help="Margin (mm) quanh bbox răng — tự convert sang voxels")
    parser.add_argument("--min_voxels", type=int, default=5000,
                        help="Min voxels để coi là 1 răng")
    parser.add_argument("--percentile_threshold", type=float, default=60.0,
                        help="Percentile ngưỡng intensity detect răng")

    return parser.parse_args()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
