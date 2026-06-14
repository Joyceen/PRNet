"""Extra modules for PRNet experiments."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqGate(nn.Module):
    """Frequency-domain gate: FFT → learned gating → IFFT."""
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, max(8, channels // 16), 1, bias=True),
            nn.BatchNorm2d(max(8, channels // 16)),
            nn.ReLU(True),
            nn.Conv2d(max(8, channels // 16), channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_fft = torch.fft.fft2(x.float())
        gate = self.gate(x_fft.real)
        x_filtered = torch.abs(torch.fft.ifft2(gate * x_fft))
        return x + x_filtered

class ConvBR(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class TextureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBR(3, 64, k=7, s=2, p=3)
        self.conv2 = ConvBR(64, 64, k=3, s=2, p=1)
        self.conv3 = ConvBR(64, 32, k=3, s=2, p=1)
        self.conv_out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        pg = self.conv_out(feat)
        return feat, pg


class GradientInjection(nn.Module):
    def __init__(self, rgb_ch, grad_ch=32, out_ch=64):
        super().__init__()
        self.fuse = ConvBR(rgb_ch + grad_ch, out_ch, k=3, s=1, p=1)

    def forward(self, x_rgb, x_grad):
        if x_grad.shape[-2:] != x_rgb.shape[-2:]:
            x_grad = F.interpolate(x_grad, size=x_rgb.shape[-2:], mode='bilinear', align_corners=True)
        return self.fuse(torch.cat([x_rgb, x_grad], dim=1))
