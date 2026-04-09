"""
Inference pipeline for CBCT tooth & root canal segmentation.

Usage:
    python inference.py --checkpoint ./checkpoints/best_model.pth --input_dir ./data/test_images --output_dir ./predictions
"""
import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AddChannel,
    Compose,
    EnsureType,
    LoadImage,
    NormalizeIntensity,
    Orientation,
    Spacing,
)

from config import DataConfig, InferenceConfig, ModelConfig
from model import build_model


class Predictor:
    def __init__(self, config: InferenceConfig, model_config: ModelConfig, data_config: DataConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = build_model(model_config, img_size=config.patch_size)
        checkpoint = torch.load(config.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {config.checkpoint_path}")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best val dice: {checkpoint.get('best_val_dice', 'unknown')}")

        # Preprocessing (same as validation)
        self.preprocess = Compose([
            LoadImage(image_only=True),
            AddChannel(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=data_config.spacing, mode="bilinear"),
            NormalizeIntensity(nonzero=True, channel_wise=True),
            EnsureType(),
        ])

    @torch.no_grad()
    def predict_volume(self, image_path: str) -> np.ndarray:
        """Run inference on a single CBCT volume."""
        # Preprocess
        image = self.preprocess(image_path)
        image = image.unsqueeze(0).to(self.device)  # (1, 1, D, H, W)

        # Sliding window inference
        output = sliding_window_inference(
            image,
            roi_size=self.config.patch_size,
            sw_batch_size=4,
            predictor=self.model,
            overlap=self.config.overlap,
            mode="gaussian",
        )

        # Test-time augmentation: flip and average
        if self.config.use_tta:
            # Horizontal flip
            flipped = torch.flip(image, dims=[4])
            output_flip = sliding_window_inference(
                flipped,
                roi_size=self.config.patch_size,
                sw_batch_size=4,
                predictor=self.model,
                overlap=self.config.overlap,
                mode="gaussian",
            )
            output_flip = torch.flip(output_flip, dims=[4])
            output = (output + output_flip) / 2.0

        # Argmax to get class labels
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return pred

    def predict_and_save(self, image_path: str, output_path: str):
        """Predict and save result as NIfTI."""
        pred = self.predict_volume(image_path)

        # Load original for header/affine info
        original = nib.load(image_path)
        pred_nii = nib.Nifti1Image(pred, affine=original.affine, header=original.header)
        nib.save(pred_nii, output_path)
        print(f"  Saved: {output_path}")

        # Print stats
        unique, counts = np.unique(pred, return_counts=True)
        total = pred.size
        for label, count in zip(unique, counts):
            label_names = {0: "background", 1: "tooth", 2: "root_canal"}
            name = label_names.get(label, f"class_{label}")
            print(f"    {name}: {count:,} voxels ({100*count/total:.2f}%)")

    def run_batch(self, input_dir: str, output_dir: str):
        """Run inference on all volumes in a directory."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        input_path = Path(input_dir)

        files = sorted(
            list(input_path.glob("*.nii.gz"))
            + list(input_path.glob("*.nii"))
            + list(input_path.glob("*.nrrd"))
        )

        if not files:
            print(f"No volumes found in {input_dir}")
            return

        print(f"\nRunning inference on {len(files)} volumes...")
        for i, filepath in enumerate(files):
            print(f"\n[{i+1}/{len(files)}] Processing: {filepath.name}")
            out_name = filepath.name.replace(".nrrd", ".nii.gz")
            if not out_name.endswith(".nii.gz"):
                out_name = filepath.stem + "_pred.nii.gz"
            else:
                out_name = filepath.stem.replace(".nii", "") + "_pred.nii.gz"
            output_path = os.path.join(output_dir, out_name)
            self.predict_and_save(str(filepath), output_path)

        print(f"\nAll predictions saved to: {output_dir}")


def find_best_kfold_checkpoint(kfold_dir: str) -> str:
    """
    Tìm checkpoint tốt nhất trong 1 thư mục K-fold dựa trên kfold_summary.json.

    Args:
        kfold_dir: thư mục chứa kfold_summary.json và các fold{N}/best_model.pth
                   (ví dụ: ./checkpoints/kfold_tooth_canal)

    Returns:
        Đường dẫn tới best_model.pth của fold có val dice cao nhất.
    """
    kfold_dir = Path(kfold_dir)
    summary_path = kfold_dir / "kfold_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {summary_path}. "
            f"Hãy chạy train_kfold.py trước hoặc truyền trực tiếp --checkpoint."
        )

    with open(summary_path) as f:
        summary = json.load(f)

    folds = summary["folds"]
    best_fold = max(folds, key=lambda r: r["best_val_dice"])
    best_ckpt = kfold_dir / f"fold{best_fold['fold']}" / "best_model.pth"

    print(f"\n[K-fold] Chọn fold tốt nhất:")
    for r in folds:
        marker = " <-- BEST" if r["fold"] == best_fold["fold"] else ""
        print(f"  Fold {r['fold']}: dice={r['best_val_dice']:.4f} "
              f"(val={r['val_cases']}){marker}")
    print(f"  -> Dùng: {best_ckpt}")

    if not best_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint không tồn tại: {best_ckpt}")

    return str(best_ckpt)


def parse_args():
    parser = argparse.ArgumentParser(description="CBCT Tooth & Canal Segmentation Inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Đường dẫn checkpoint .pth. "
                             "Nếu không truyền, sẽ dùng --kfold_dir để tự chọn fold tốt nhất.")
    parser.add_argument("--kfold_dir", type=str, default=None,
                        help="Thư mục K-fold experiment (chứa kfold_summary.json). "
                             "Sẽ tự động chọn fold có val dice cao nhất.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input volumes")
    parser.add_argument("--output_dir", type=str, default="./predictions")
    parser.add_argument("--arch", type=str, default="nnunet",
                        choices=["nnunet", "unet3d", "swin_unetr"])
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--no_tta", action="store_true", help="Disable test-time augmentation")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve checkpoint path: ưu tiên --checkpoint, fallback sang --kfold_dir
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.kfold_dir:
        checkpoint_path = find_best_kfold_checkpoint(args.kfold_dir)
    else:
        raise ValueError(
            "Cần truyền --checkpoint <path> HOẶC --kfold_dir <kfold_experiment_dir>"
        )

    inference_config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        patch_size=tuple(args.patch_size),
        overlap=args.overlap,
        use_tta=not args.no_tta,
    )
    model_config = ModelConfig(architecture=args.arch)
    data_config = DataConfig()

    predictor = Predictor(inference_config, model_config, data_config)
    predictor.run_batch(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
