import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):  # convlution 2 times in this layer

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
# in_c --> out_c


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # pooling the figure make the size smaller
            DoubleConv(in_channels, out_channels) # convlution 2 times
        )

    def forward(self, x):
        return self.maxpool_conv(x)
# -->Bx in_C x H x W --> B x out_c x H/2 x W/2


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # double the figure size
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2) #convolution the image 2 times, and make channels larger

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1) # copy and crop from the corresponding down layer
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=32, dropout_ratio=0.1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 512)
        self.down2 = Down(512, 1024)
        self.down3 = Down(1024, 2048)
        self.down4 = Down(2048, 2048)

        self.up1 = Up(4096, 1024)
        self.up2 = Up(2048, 512)
        self.up3 = Up(1024, 64)
        self.up4 = Up(128, 64)
        self.drop = nn.Dropout(dropout_ratio)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.drop(x1)
        #--> B x 64 x 720 x 960
        x2 = self.down1(x1)
        x2 = self.drop(x2)
        #--> B x 256 x 360 x 480
        x3 = self.down2(x2)
        x3 = self.drop(x3)
        #--> B x 1024 x 180 x 240
        x4 = self.down3(x3)
        x4 = self.drop(x4)
        #--> B x 2048 x 90 x 120
        x5 = self.down4(x4)
        x5 = self.drop(x5)
        #--> B x 2048 x 45 x 60

        x = self.up1(x5, x4)
        x = self.drop(x)
        #--> B x 1024 x 90 x 120
        x = self.up2(x, x3)
        x = self.drop(x)
        #--> B x 512 x 180 x 240
        x = self.up3(x, x2)
        x = self.drop(x)
        #--> B x 256 x 360 x 480
        x = self.up4(x, x1)
        x = self.drop(x)
        #--> B x 128 x 720 x 960
        logits = self.outc(x)
        #--> B x 32 x 720 x 960
        return logits