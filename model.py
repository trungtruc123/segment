"""
Model architectures for CBCT tooth & root canal segmentation.
Supports: 3D UNet, Swin-UNETR (via MONAI).
"""
from typing import List, Tuple

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet, SwinUNETR

from config import ModelConfig


class ConvBlock3D(nn.Module):
    """Double convolution block: Conv3d -> BN -> ReLU -> Conv3d -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """
    3D U-Net with instance normalization and deep supervision.
    Inspired by nnU-Net architecture choices.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        features: List[int] = None,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder path
        for i, feat in enumerate(features):
            in_ch = in_channels if i == 0 else features[i - 1]
            self.encoders.append(ConvBlock3D(in_ch, feat))
            if i < len(features) - 1:
                self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))

        # Decoder path
        for i in range(len(features) - 2, -1, -1):
            self.upconvs.append(
                nn.ConvTranspose3d(features[i + 1], features[i], kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock3D(features[i] * 2, features[i]))

        # Deep supervision outputs
        self.deep_outputs = nn.ModuleList()
        for i in range(min(3, len(features) - 1)):
            self.deep_outputs.append(nn.Conv3d(features[i], num_classes, kernel_size=1))

        self.final_conv = nn.Conv3d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connections = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < len(self.pools):
                skip_connections.append(x)
                x = self.pools[i](x)

        # Decoder
        deep_outputs = []
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skip_connections[-(i + 1)]
            # Handle size mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

            # Deep supervision (collect from lowest decoder levels)
            dec_idx = len(self.upconvs) - 1 - i
            if dec_idx < len(self.deep_outputs):
                deep_outputs.append(self.deep_outputs[dec_idx](x))

        output = self.final_conv(x)

        if self.training and deep_outputs:
            return output, deep_outputs
        return output


def build_nnunet(
    config: ModelConfig,
    img_size: Tuple[int, int, int] = (96, 96, 96),
) -> DynUNet:
    """
    Build nnU-Net architecture via MONAI's DynUNet.

    DynUNet là implementation chính thức của nnU-Net trong MONAI:
        - Instance normalization
        - LeakyReLU (slope 0.01)
        - Deep supervision ở nhiều decoder levels
        - Không có residual (standard nnUNet)
        - Strided conv cho downsampling, transposed conv cho upsampling

    Với patch 96x96x96, dùng 5 levels downsampling (96 -> 48 -> 24 -> 12 -> 6 -> 3).
    """
    # Kernel + stride theo công thức của nnU-Net:
    #   Level 0: không downsample
    #   Các level sau: stride=2 nếu spatial dim còn chia được cho 2
    kernel_size = [[3, 3, 3]] * 6
    strides = [
        [1, 1, 1],  # level 0
        [2, 2, 2],  # level 1: 96 -> 48
        [2, 2, 2],  # level 2: 48 -> 24
        [2, 2, 2],  # level 3: 24 -> 12
        [2, 2, 2],  # level 4: 12 -> 6
        [2, 2, 2],  # level 5: 6 -> 3 (bottleneck)
    ]
    upsample_kernel_size = strides[1:]  # DynUNet yêu cầu

    model = DynUNet(
        spatial_dims=3,
        in_channels=config.in_channels,
        out_channels=config.num_classes,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_kernel_size,
        norm_name="instance",
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision=True,
        deep_supr_num=2,         # supervise 2 decoder levels ngoài output chính
        res_block=False,         # standard nnUNet, không dùng residual
        trans_bias=False,
    )
    return model


def build_swin_unetr(config: ModelConfig, img_size: Tuple[int, int, int] = (96, 96, 96)) -> SwinUNETR:
    """Build Swin-UNETR model from MONAI."""
    model = SwinUNETR(
        img_size=img_size,
        in_channels=config.in_channels,
        out_channels=config.num_classes,
        feature_size=config.swin_feature_size,
        depths=config.swin_depths,
        num_heads=config.swin_num_heads,
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=True,  # gradient checkpointing to save memory
        spatial_dims=3,
    )
    return model


def build_model(config: ModelConfig, img_size: Tuple[int, int, int] = (96, 96, 96)) -> nn.Module:
    """Factory function to build the selected model."""
    if config.architecture == "unet3d":
        model = UNet3D(
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            features=config.unet_features,
        )
    elif config.architecture == "nnunet":
        model = build_nnunet(config, img_size)
    elif config.architecture == "swin_unetr":
        model = build_swin_unetr(config, img_size)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config.architecture}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    return model
