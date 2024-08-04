"""Implementation of 3D U-Net with AdaIN normalization for resolution-agnostic image processing.

Refs:
    - https://pubmed.ncbi.nlm.nih.gov/35605505/
    - https://github.com/YSerin/TMI_SwitchableCycleGAN/blob/train/networks/adain.py
    - https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""

import torch
from torch import nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    """Adaptive instance normalization layer with code generator."""

    def __init__(self, feat_channels, shared_adain_code_dim):
        super().__init__()

        self.norm = nn.InstanceNorm3d(feat_channels)
        self.fc_mean = nn.Linear(shared_adain_code_dim, feat_channels)
        self.fc_std = nn.Linear(shared_adain_code_dim, feat_channels)

    def forward(self, x, shared_adain_code):
        mean = self.fc_mean(shared_adain_code)
        std = F.relu(self.fc_std(shared_adain_code))

        b, c, d, h, w = x.size()
        x = self.norm(x)
        x = x * std.view(b, c, 1, 1, 1) + mean.view(b, c, 1, 1, 1)
        return x


class ConvAdaIN(nn.Module):
    """Convolutional layer, then AdaIN normalization, then LeadkyRelu."""

    def __init__(self, in_channels, out_channels, shared_adain_code_dim):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.adain = AdaIN(out_channels, shared_adain_code_dim)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x, shared_adain_code):
        x = self.conv(x)
        x = self.adain(x, shared_adain_code)
        x = self.lrelu(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shared_adain_code_dim):
        super().__init__()

        self.down = nn.Conv3d(in_channels, in_channels, 3, stride=2, padding=1)
        self.conv_adain = ConvAdaIN(in_channels, out_channels, shared_adain_code_dim)

    def forward(self, x, shared_adain_code):
        x = self.down(x)
        x = self.conv_adain(x, shared_adain_code)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shared_adain_code_dim):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv_adain = ConvAdaIN(
            out_channels * 2, out_channels, shared_adain_code_dim
        )

    def forward(self, x, x_down, shared_adain_code):
        x = self.up_conv(self.up(x))
        x = torch.cat([x, x_down], dim=1)
        x = self.conv_adain(x, shared_adain_code)
        return x


class AdaINCodeGeneratorShared(nn.Module):
    def __init__(self, in_dim, shared_adain_code_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, shared_adain_code_dim)
        self.fc2 = nn.Linear(shared_adain_code_dim, shared_adain_code_dim)
        self.fc3 = nn.Linear(shared_adain_code_dim, shared_adain_code_dim)
        self.fc4 = nn.Linear(shared_adain_code_dim, shared_adain_code_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        return x


class UNetAdaIN3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        shared_adain_code_dim = 128
        feat = 32  # base feature dimension

        self.in_conv = ConvAdaIN(in_channels, feat, shared_adain_code_dim)
        self.down1 = DownBlock(feat, feat * 2, shared_adain_code_dim)
        self.down2 = DownBlock(feat * 2, feat * 4, shared_adain_code_dim)
        self.down3 = DownBlock(feat * 4, feat * 8, shared_adain_code_dim)

        self.up1 = UpBlock(feat * 8, feat * 4, shared_adain_code_dim)
        self.up2 = UpBlock(feat * 4, feat * 2, shared_adain_code_dim)
        self.up3 = UpBlock(feat * 2, feat, shared_adain_code_dim)

        self.out_conv = nn.Conv3d(feat, out_channels, 1)

        self.adain_code_generator_shared = AdaINCodeGeneratorShared(
            3, shared_adain_code_dim
        )

    def forward(self, x, input_adain_code):
        """
        Args:
            x: (B, C, D, H, W)
            input_adain_code: (B, 3), voxel size in mm
        """
        shared_adain_code = self.adain_code_generator_shared(input_adain_code)
        x1 = self.in_conv(x, shared_adain_code)
        x2 = self.down1(x1, shared_adain_code)
        x3 = self.down2(x2, shared_adain_code)
        x4 = self.down3(x3, shared_adain_code)
        x = self.up1(x4, x3, shared_adain_code)
        x = self.up2(x, x2, shared_adain_code)
        x = self.up3(x, x1, shared_adain_code)
        x = self.out_conv(x)
        return x


def test_unet_adain_3d():
    B, C, D, H, W = 1, 1, 64, 64, 64
    x = torch.randn(B, C, D, H, W)
    input_adain_code = torch.randn(B, 3)
    model = UNetAdaIN3D(in_channels=C, out_channels=C)
    y = model(x, input_adain_code)
    assert y.shape == (B, C, D, H, W)
    print("test passed")


if __name__ == "__main__":
    test_unet_adain_3d()
