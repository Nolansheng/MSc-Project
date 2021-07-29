import torch
import torch.nn as nn


#ResNet
# one of the minimum convolution block use a 3x3 kernel
# doesn't affect image size
def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    )


# one of the minimum convolution block use a 1x1 kernel
# doesn't affect image size channels from in_planes becomes out_planes
def conv1x1(in_planes, out_planes, kernel_size=1, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# def block, 1x1 convolution + 3x3 convolution + 1x1 convolution
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # HxWxinplanes
        self.conv1 = conv1x1(inplanes, outplanes, kernel_size=1, stride=stride)
        self.bn1 = norm_layer(outplanes)

        # self.conv2 = conv3x3(outplanes, outplanes, stride)
        self.conv2 = conv3x3(outplanes, outplanes)
        self.bn2 = norm_layer(outplanes)

        self.conv3 = conv1x1(outplanes, outplanes*self.expansion, kernel_size=1, stride=1)
        self.bn3 = norm_layer(outplanes*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # in HxWxinplanes
        out = self.conv1(x)
        # out HxWxoutplanes
        out = self.bn1(out)
        # out HxWxoutplanes
        out = self.relu(out)
        # out HxWxoutplanes

        # in HxWxoutplanes
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # out HxWxoutplanes

        # in HxWxoutplanes
        out = self.conv3(out)
        out = self.bn3(out)
        # out HxWx(outplanes*4)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity # shortcut connection and element-wise addition F+x
        out = self.relu(out)
        # out HxWx(outplanes*4)
        return out





class ResNet(nn.Module):
    def __init__(self,  block, layers, norm_layer = None, num_classes=32):
        super(ResNet, self).__init__()

        self.block = block
        self.layers = layers

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64

        # conv1
        # initial image size is 3x720x960
        # from 3 channel to 64 channel
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        # --> 64x720x960
        self.bn1=nn.BatchNorm2d(self.inplanes)
        self.relu=nn.ReLU(inplace=True)

        # max pooling layer
        self.maxpool=nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        # out 64x360x480

        self.stage1 = self.make_layer(self.block, 64, self.layers[0], stride=1)
        # out 64x360x480
        self.stage2 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        # out 128x180x240
        self.stage3 = self.make_layer(self.block, 512, self.layers[2], stride=2)
        # out 512x90x120
        self.stage4 = self.make_layer(self.block, 1024, self.layers[3], stride=2)
        # out 1024x45x60


    def make_layer(self, block, midplane, block_num, stride=1):
        # to decide whether need downsample module
        downsample = None
        # in HxWxinplane
        if (stride != 1 or self.inplanes != midplane * block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, midplane * block.expansion,
                          stride=stride,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(midplane * block.expansion)
            )
        # out HxWx(midplane*4)

        block_list = []
        block_list.append(block(self.inplanes, midplane, stride, downsample))
        self.inplanes = midplane * block.expansion
        for i in range(1, block_num):
            block_list.append(block(self.inplanes, midplane, stride=1))

        return nn.Sequential(*block_list)

        # for the first time, input is 64 channel, but output is 256,
        # then use downsample make the identity becomes 256 channel
        # then add the output with identity
        # for stage1 (64->64->256)=>(64->64->256)=>(64->64=>256) after this self.inplanes = 256
        # for stage2 256=>(128->128->512)=>(128->128->512)=>......(128->128->512) after this self.inplanes = 512

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # block
        out = self.stage1(out)
        out = self.stage2(out)
        # # stage2 = out
        out = self.stage3(out)
        # # stage3 = out
        out = self.stage4(out)
        # stage4 = out
        # output is batch*2048*3023
        return out


def resnet50():
    layer50 = [3, 4, 6, 3]
    model = ResNet(Bottleneck, layer50)
    return model


def resnet101():
    layer101 = [3, 4, 23, 3]
    model = ResNet(Bottleneck, layer101)
    return model


def resnet34():
    layer34 = [3, 4, 6, 3]
    model = ResNet(BasicBlock, layer34)
    return model

def resnet18():
    layer18 = [2, 2, 2, 2]
    model = ResNet(BasicBlock, layer18)
    return model

