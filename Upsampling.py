import torch
from torch import nn
import torch.nn.functional as F
import ResNet as RESNET

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


class Upsample(nn.Module):
    def __init__(self, layers=50, classes=32):
        super(Upsample, self).__init__()
        if layers == 50:
            resnet = RESNET.resnet50()
        if layers == 101:
            resnet = RESNET.resnet101()
        # output from resnet is [batch_size, 2048, 23, 30]
        self.layer0 = resnet
        self.up1 = Up([45, 60], 2048, 1024)

        self.up2 = Up([180, 240], 1024, 256)

        self.up3 = Up([720, 960], 256, classes)

    def forward(self, x):
        x = self.layer0(x)
        # --> batch*2048x23x30
        x = self.up1(x)
        # --> batch*1024x45x60
        x = self.up2(x)
        # --> batch*256x180x240
        x = self.up3(x)
        # --> batch*32x720x960
        return x