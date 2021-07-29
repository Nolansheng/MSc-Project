import torch
import torch.nn as nn
import numpy as np
import ResNet as RESNET
from Transformer import Transformer
import torch.nn.functional as F

class PPM(nn.Module):
    def __init__(self, in_dim=1024, reduction_dim=512, bins=(2, 3, 5, 8)):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin), # pooling get 2x2, 3x3, 6x6, 11x11
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()

        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


# class Up(nn.Module):
#     def __init__(self, out_size, in_channels, out_channels):
#         super(Up, self).__init__()
#         self.up = nn.Upsample(out_size, mode='bilinear', align_corners=True)
#         self.conv = DoubleConv(in_channels, out_channels, in_channels//2)
#
#     def forward(self, x):
#         x = self.up(x)
#         x = self.conv(x)
#
#         return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # double the figure size
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2) #convolution the image 2 times, and make channels larger

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1) # copy and crop from the corresponding down layer
        return self.conv(x)


class DoubleConv(nn.Module):  # Defines a double convolutional layer to be used in the network.
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class Hybrid_model(nn.Module):
    def __init__(self, layers=18, bins=(2, 3, 5, 8), classes=32, dropout=0.1):
        super(Hybrid_model, self).__init__()
        if layers == 18:
            resnet = RESNET.resnet18()
        elif layers == 34:
            resnet = RESNET.resnet34()
        elif layers == 50:
            resnet = RESNET.resnet50()
        else:
            resnet = RESNET.resnet101()
        # output from resnet is [batch_size, 2048, 30, 23]
        self.layer0 = resnet

        self.transformer_layer = Transformer()

        fea_dim = 512
        self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins)
        fea_dim *= 2

        self.up1 = Up(2048, 512)
        self.up2 = Up(1024, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, classes)
        self.pooling = nn.MaxPool2d(2)
        self.drop = nn.Dropout(dropout)


    def forward(self, x):
        # -->batch_size x 3 x 720 x 960
        x, stage1, stage2, stage3, stage4 = self.layer0(x)
        # x = B x 1024 x 45 x 60
        # stage4 = B x 1024 x 45 x 60
        # stage3 = B x 512 x 90 x 120
        # stage2 = B x 128 x 180 x 240
        # stage1 = B x 64 x 360 x 480
        x = self.pooling(x)
        # x = B x 1024 x 23 x 30
        x = self.transformer_layer(x)
        # x = B x 512 x 23 x 30
        x = self.ppm(x)
        # x = B x 1024 x 23 x 30
        x = self.up1(x, stage4)
        x = self.drop(x)
        # x = B x 512 x 45 x 60
        x = self.up2(x, stage3)
        x = self.drop(x)
        # x = B x 128 x 90 x 120
        x = self.up3(x, stage2)
        x = self.drop(x)
        # x = B x 64 x 180 x 240
        x = self.up4(x, stage1)
        x = self.drop(x)
        # x = B x 64 x 360 x 480
        x = self.outc(x)
        # x = B x 32 x 720 x 960
        return x
