import torch
from torch import nn
import torch.nn.functional as F
import ResNet as RESNET


class PPM(nn.Module):
    def __init__(self, in_dim=2048, reduction_dim=512, bins=(1, 2, 3, 6)):
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


class Up(nn.Module):
    def __init__(self, out_size, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(out_size, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels//2)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)

        return x


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


class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), classes=32):
        super(PSPNet, self).__init__()
        if layers == 50:
            resnet = RESNET.resnet50()
        if layers == 101:
            resnet = RESNET.resnet101()
        # output from resnet is [batch_size, 2048, 30, 23]
        self.layer0 = resnet

        self.ppm = PPM(bins=bins)

        self.up1 = Up([45, 60], 4096, 1024)

        self.up2 = Up([180, 240], 1024, 256)

        self.up3 = Up([720, 960], 256, classes)

    def forward(self, x):

        # --> batch*3x720x960
        x = self.layer0(x)
        # --> batch*2048x23x30
        x = self.ppm(x)
        # --> batch*4096x23x30
        x = self.up1(x)
        # --> batch*1024x45x60
        x = self.up2(x)
        # --> batch*256x180x240
        x = self.up3(x)
        # --> batch*32x720x960

        return x

