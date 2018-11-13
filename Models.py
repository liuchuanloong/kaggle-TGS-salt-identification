import ResNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from Network import SegmentationNetwork

class ELU_1(nn.ELU):
    def __init__(self, *args, **kwargs):
        super(ELU_1, self).__init__(*args, **kwargs)

    def forward(self, input):
        return F.elu(input, self.alpha, self.inplace)

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1), groups=1, dilation=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False,
                              groups=groups,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SpatialGate2d(nn.Module):

    def __init__(self, in_channels):
        super(SpatialGate2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cal = self.conv1(x)
        cal = self.sigmoid(cal)
        return cal * x

class ChannelGate2d(nn.Module):

    def __init__(self, channels, reduction=2):
        super(ChannelGate2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x

class scSqueezeExcitationGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super(scSqueezeExcitationGate, self).__init__()
        self.spatial_gate = SpatialGate2d(channels)
        self.channel_gate = ChannelGate2d(channels, reduction=reduction)

    def  forward(self, x, z=None):
        XsSE = self.spatial_gate(x)
        XcSe = self.channel_gate(x)
        return XsSE + XcSe

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels, convT_channels=0, convT_ratio=2, SE=False, activation=None):
        super(Decoder, self).__init__()
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

        self.SE = SE
        self.convT_channels = convT_channels
        self.conv1 = ConvBn2d(in_channels, channels,
                              kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels,
                              kernel_size=3, padding=1)
        if convT_channels:
            self.convT = nn.ConvTranspose2d(convT_channels, convT_channels // convT_ratio, kernel_size=2, stride=2)

        if SE:
            self.scSE = scSqueezeExcitationGate(out_channels)

    def forward(self, x, skip):
        if self.convT_channels:
            x = self.convT(x)
        else:
            x = F.upsample(x, scale_factor=2, mode='bilinear',
                           align_corners=True)  # False
        x = torch.cat([x, skip], 1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        if self.SE:
            x = self.scSE(x)

        return x

class PyramidPoolingModule(nn.Module):
    def __init__(self, pool_list, in_channels, size=(128, 128)):
        super(PyramidPoolingModule, self).__init__()
        self.size = size
        self.pool_list = pool_list
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0) for _ in range(len(pool_list))])
        self.conv2 = nn.Conv2d(in_channels + len(pool_list), in_channels, kernel_size=1)

    def forward(self, x):
        cat = [x]
        for (k, s), conv in zip(self.pool_list, self.conv1):
            out = F.avg_pool2d(x, kernel_size=k, stride=s)
            out = conv(out)
            out = F.upsample(out, size=self.size, mode='bilinear', align_corners=True)
            cat.append(out)
        out = torch.cat(cat, 1)
        out = self.conv2(out)
        out = self.relu(out)
        return out


class SppBlock(nn.Module):
    def __init__(self, level, in_channel=256):
        super().__init__()
        self.level = level
        self.convblock = nn.Sequential(nn.Conv2d(in_channel, 512, 1, 1),
                                       nn.BatchNorm2d(512), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]
        x = F.adaptive_avg_pool2d(x, output_size=(self.level, self.level))  # average pool
        x = self.convblock(x)
        x = F.upsample(x, size=size, mode='bilinear', align_corners=True)
        return x


class SPP(nn.Module):
    def __init__(self, in_channel=256):
        super().__init__()
        self.spp1 = SppBlock(level=1, in_channel=in_channel)
        self.spp2 = SppBlock(level=2, in_channel=in_channel)
        self.spp3 = SppBlock(level=3, in_channel=in_channel)
        self.spp6 = SppBlock(level=6, in_channel=in_channel)

    def forward(self, x):
        x1 = self.spp1(x)
        x2 = self.spp2(x)
        x3 = self.spp3(x)
        x6 = self.spp6(x)
        out = torch.cat([x, x1, x2, x3, x6], dim=1)
        return out

class ExtractPyramidFeatures(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExtractPyramidFeatures, self).__init__()
        self.conv1 = nn.ModuleList(
            [ConvBn2d(in_c, out_channels, kernel_size=1, padding=0) for in_c in in_channels])
        self.conv2 = nn.ModuleList(
            [ConvBn2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in range(len(in_channels)-1)])

    def forward(self, C2, C3, C4, C5):
        P5 = self.conv1[-1](C5)
        P5_up = F.upsample(P5, scale_factor=2, mode='bilinear', align_corners=True)

        P4 = self.conv1[-2](C4)
        P4 = P4 + P5_up
        P4 = self.conv2[0](P4)
        P4_up = F.upsample(P4, scale_factor=2, mode='bilinear', align_corners=True)

        P3 = self.conv1[-3](C3)
        P3 = P3 + P4_up
        P3 = self.conv2[1](P3)
        P3_up = F.upsample(P3, scale_factor=2, mode='bilinear', align_corners=True)

        P2 = self.conv1[-4](C2)
        P2 = P2 + P3_up
        P2 = self.conv2[2](P2)

        return P2, P3, P4, P5

class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Conv2dBn(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBn, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.conv(x)

class GAUModule(nn.Module):
    def __init__(self, in_ch, out_ch):  #
        super(GAUModule, self).__init__()

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBn(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.conv2 = Conv2dBnRelu(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    # x: low level feature
    # y: high level feature
    def forward(self, y, x):
        h, w = x.size(2), x.size(3)
        y_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y)
        x = self.conv2(x)
        y = self.conv1(y)
        z = torch.mul(x, y)

        return y_up + z

class FPAModule(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(FPAModule, self).__init__()

        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        # midddle branch
        self.mid = nn.Sequential(
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(in_ch, out_ch, kernel_size=7, stride=1, padding=3)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            Conv2dBnRelu(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

        self.conv2 = Conv2dBnRelu(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(out_ch, out_ch, kernel_size=7, stride=1, padding=3)

        self.scSE = scSqueezeExcitationGate(out_ch)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        # b1 = self.scSE(b1)

        b1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(b1)

        mid = self.mid(x)
        # mid = self.scSE(mid)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # x3 = self.scSE(x3)
        x3 = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3)

        x2 = self.conv2(x2)
        # x2 = self.scSE(x2)
        x = x2 + x3
        x = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x)

        x1 = self.conv1(x1)
        # x1 = self.scSE(x1)
        x = x + x1
        x = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x)

        x = torch.mul(x, mid)
        x = x + b1
        return x

class PyramidPrediction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PyramidPrediction, self).__init__()
        self.conv1 = ConvBn2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, upsampling=None):
        x = self.conv1(x)
        x = self.conv2(x)
        if upsampling:
            x = F.upsample(x, scale_factor=upsampling, mode='bilinear', align_corners=True)
        return  x

class ResidualConvUnit(nn.Module):
    def __init__(self, channels, SE=False, bottleneck=1):
        super(ResidualConvUnit, self).__init__()
        if bottleneck > 1:
            c = channels // bottleneck
            self.convb1 = ConvBn2d(channels, c, kernel_size=1, padding=0)
            self.convb2 = ConvBn2d(c, channels, kernel_size=1, padding=0)

        self.SE = SE
        self.bottleneck = bottleneck
        self.relu = nn.ReLU(inplace=False)
        c = channels // bottleneck
        self.conv1 = ConvBn2d(c, c, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(c, c, kernel_size=3, padding=1)

        if SE:
            self.se = scSqueezeExcitationGate(channels)

    def forward(self, x):
        residual = x
        if self.bottleneck > 1:
            x = self.convb1(x)
            x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.bottleneck > 1:
            x = self.convb2(x)
            x = self.relu(x)
        if self.SE:
            x = self.se(x)

        x += residual
        return self.relu(x)

class BottleNeckResidualConvUnit(nn.Module):
    def __init__(self, channels, SE=False, bottleneck=4):
        super(BottleNeckResidualConvUnit, self).__init__()
        c = channels // bottleneck
        self.convb1 = ConvBn2d(channels, c, kernel_size=1, padding=0)
        self.convb2 = ConvBn2d(c, channels, kernel_size=1, padding=0)

        self.SE = SE
        self.relu = nn.ReLU(inplace=False)

        self.conv1 = ConvBn2d(c, c, kernel_size=3, padding=1)

        if SE:
            self.se = scSqueezeExcitationGate(channels)

    def forward(self, x):
        residual = x
        x = self.convb1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.convb2(x)
        if self.SE:
            x = self.se(x)

        x += residual
        return self.relu(x)

class MultiResolutionFusion(nn.Module):
    def __init__(self, channels, skip_in=256):
        super(MultiResolutionFusion, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if skip_in is not None:
            self.conv2 = nn.Conv2d(skip_in, channels, kernel_size=3, padding=1)

    def forward(self, x, skip=None):
        x = self.conv1(x)
        if skip is not None:
            skip = self.conv2(skip)
            skip = F.upsample(skip, scale_factor=2, mode='bilinear', align_corners=False)
            return x + skip
        else:
            return x

class ChainedResidualPooling(nn.Module):
    def __init__(self, channels, pool_mode='avg', SE=False):
        super(ChainedResidualPooling, self).__init__()
        self.SE = SE
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = ConvBn2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, channels, kernel_size=3, padding=1)
        if pool_mode == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        elif pool_mode == 'max':
            self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        if SE:
            self.se = scSqueezeExcitationGate(channels)

    def forward(self, x):
        residual1 = x
        x = self.pool(x)
        x = self.conv1(x)
        residual2 = x
        x = self.pool(x)
        x = self.conv2(x)
        if self.SE:
            x = self.se(x)
        x += residual2 + residual1
        x = self.relu(x)
        return x

class RefineNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_in=None, pool_mode='avg', SE=False):
        super(RefineNetBlock, self).__init__()
        self.SE = SE
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.residual1 = ResidualConvUnit(out_channels)
        self.residual2 = ResidualConvUnit(out_channels)
        self.fusion = MultiResolutionFusion(out_channels, skip_in)
        self.pool = ChainedResidualPooling(out_channels, pool_mode=pool_mode)
        self.residual3 = ResidualConvUnit(out_channels)
        if SE:
            self.scSE = scSqueezeExcitationGate(out_channels)


    def forward(self, x, skip=None):
        x = self.conv1(x)
        long_residual = x
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.fusion(x, skip)
        x = self.pool(x)
        x = self.residual3(x)
        x = x + long_residual
        if self.SE:
            x = self.scSE(x)
        return x

class Decoder_v2(nn.Module):
    def __init__(self, in_channels, channels, out_channels, convT_channels=0, convT_ratio=2, SE=False, activation=None):
        super(Decoder_v2, self).__init__()
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

        self.SE = SE
        self.convT_channels = convT_channels

        self.conv1 = ConvBn2d(in_channels, channels,
                              kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels,
                              kernel_size=3, padding=1)

        if convT_channels:
            self.convT = nn.ConvTranspose2d(convT_channels, convT_channels // convT_ratio, kernel_size=2, stride=2)

        if SE:
            self.scSE = scSqueezeExcitationGate(out_channels)

        self.conv_res = nn.Conv2d(convT_channels // convT_ratio, out_channels, kernel_size=1, padding=0)

    def forward(self, x, skip, z=None):
        if self.convT_channels:
            x = self.convT(x)
            x = self.activation(x)
        else:
            x = F.upsample(x, scale_factor=2, mode='bilinear',
                           align_corners=True)  # False

        residual = x

        x = torch.cat([x, skip], 1)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)

        if self.SE:
            x = self.scSE(x)

        x += self.conv_res(residual)
        x = self.activation(x)
        return x

class BottleneckDecoder_v2(nn.Module):
    def __init__(self, in_channels, out_channels, convT_channels, bottleneck=4, SE=False):
        super(BottleneckDecoder_v2, self).__init__()
        c = out_channels // bottleneck
        self.convb1 = ConvBn2d(in_channels, c, kernel_size=1, padding=0)
        self.convb2 = ConvBn2d(c, out_channels, kernel_size=1, padding=0)
        self.convbT = ConvBn2d(convT_channels, c, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.SE = SE

        self.conv1 = ConvBn2d(2 * c, c,
                              kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(c, c,
                              kernel_size=3, padding=1)

        self.convT = nn.ConvTranspose2d(c, c, kernel_size=2, stride=2)

        if SE:
            self.scSE = scSqueezeExcitationGate(out_channels)

        self.conv_res = nn.Conv2d(c, out_channels, kernel_size=1, padding=0)

    def forward(self, x, skip):
        x = self.convbT(x)
        x = self.relu(x)
        x = self.convT(x)
        x = self.relu(x)

        residual = x

        skip = self.convb1(skip)
        skip = self.relu(skip)
        x = torch.cat([x, skip], 1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.convb2(x)

        if self.SE:
            x = self.scSE(x)

        x += self.conv_res(residual)
        x = self.relu(x)
        return x

class CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, SE=False):
        super(CenterBlock, self).__init__()
        self.SE = SE
        self.pool = pool
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConvBn2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        if SE:
            self.se = scSqueezeExcitationGate(out_channels)

    def forward(self, x):
        if self.pool:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        residual = self.conv_res(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        if self.SE:
            x = self.se(x)

        x += residual
        x = self.relu(x)
        return x

class BottleneckCenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=4, pool=True, SE=False):
        super(BottleneckCenterBlock, self).__init__()
        self.SE = SE
        self.pool = pool

        c = out_channels // bottleneck
        self.convb1 = ConvBn2d(in_channels, c, kernel_size=1, padding=0)
        self.convb2 = ConvBn2d(c, out_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConvBn2d(c, c, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(c, c, kernel_size=3, padding=1)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        if SE:
            self.se = scSqueezeExcitationGate(out_channels)

    def forward(self, x):
        if self.pool:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        residual = self.conv_res(x)
        x = self.convb1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.convb2(x)

        if self.SE:
            x = self.se(x)

        x += residual
        x = self.relu(x)
        return x

class HyperBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=4, SE=False):
        super(HyperBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.residual = ResidualConvUnit(out_channels, bottleneck=bottleneck, SE=SE)
        self.pool = ChainedResidualPooling(out_channels, SE=SE)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.residual(x)
        x = self.pool(x)
        x += self.conv_res(residual)
        return x

class Decoder_v3(nn.Module):
    def __init__(self, in_channels, convT_channels, out_channels, convT_ratio=2, SE=False):
        super(Decoder_v3, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.SE = SE
        self.convT = nn.ConvTranspose2d(convT_channels, convT_channels // convT_ratio, kernel_size=2, stride=2)
        self.conv1 = ConvBn2d(in_channels  + convT_channels // convT_ratio, out_channels)
        self.conv2 = ConvBn2d(out_channels, out_channels)
        if SE:
            self.scSE = scSqueezeExcitationGate(out_channels)

        self.conv_res = nn.Conv2d(convT_channels // convT_ratio, out_channels, kernel_size=1, padding=0)

    def forward(self, x, skip):
        x = self.convT(x)
        residual = x
        x = torch.cat([x, skip], 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.SE:
            x = self.scSE(x)
        x += self.conv_res(residual)
        x = self.relu(x)
        return x

#######################################################################

class UNetResNet34(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, **kwargs):
        super(UNetResNet34, self).__init__(**kwargs)
        self.resnet = ResNet.resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64


        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.center = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBn2d(512, 1024,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(1024, 1024,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(512 + 512, 512, 512, convT_channels=1024)
        self.decoder4 = Decoder(256 + 256, 256, 256, convT_channels=512)
        self.decoder3 = Decoder(128 + 128, 128, 128, convT_channels=256)
        self.decoder2 = Decoder(64 + 64, 64, 64, convT_channels=128)
        self.decoder1 = Decoder(32 + 64, 64, 32, convT_channels=64)

        self.logit = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        f = self.center(e4)  # 4

        f = self.decoder5(f, e4)  # 8
        f = self.decoder4(f, e3)  # 16
        f = self.decoder3(f, e2)  # 32
        f = self.decoder2(f, e1)  # 64
        f = self.decoder1(f, x)

        # f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)  # ; print('logit',logit.size())
        return logit

class UNetResNet34_SE(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(UNetResNet34_SE, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = ELU_1(inplace=True)

        self.resnet = ResNet.resnet34(pretrained=pretrained, activation=self.activation)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.activation,
        )  # 64


        self.encoder1 = nn.Sequential(self.resnet.layer1, scSqueezeExcitationGate(64))# 64
        self.encoder2 = nn.Sequential(self.resnet.layer2, scSqueezeExcitationGate(128))# 128
        self.encoder3 = nn.Sequential(self.resnet.layer3, scSqueezeExcitationGate(256))# 256
        self.encoder4 = nn.Sequential(self.resnet.layer4, scSqueezeExcitationGate(512))# 512

        self.center = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBn2d(512, 1024,
                     kernel_size=3, padding=1),
            self.activation,
            ConvBn2d(1024, 1024,
                     kernel_size=3, padding=1),
            self.activation,
        )

        self.decoder5 = Decoder(512 + 512, 512, 512, convT_channels=1024, SE=True, activation=self.activation)
        self.decoder4 = Decoder(256 + 256, 256, 256, convT_channels=512, SE=True, activation=self.activation)
        self.decoder3 = Decoder(128 + 128, 128, 128, convT_channels=256, SE=True, activation=self.activation)
        self.decoder2 = Decoder(64 + 64, 64, 64, convT_channels=128, SE=True, activation=self.activation)
        self.decoder1 = Decoder(32 + 64, 64, 32, convT_channels=64, SE=True, activation=self.activation)

        self.logit = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        f = self.center(e4)  # 4

        f = self.decoder5(f, e4)  # 8
        f = self.decoder4(f, e3)  # 16
        f = self.decoder3(f, e2)  # 32
        f = self.decoder2(f, e1)  # 64
        f = self.decoder1(f, x)

        # f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)
        return logit

class UNetResNet34_SE_Hyper(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(UNetResNet34_SE_Hyper, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = ELU_1(inplace=True)

        self.resnet = ResNet.resnet34(pretrained=pretrained, activation=self.activation, SE=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.activation,
        )  # 64


        self.encoder1 = self.resnet.layer1# 64
        self.encoder2 = self.resnet.layer2# 128
        self.encoder3 = self.resnet.layer3# 256
        self.encoder4 = self.resnet.layer4# 512

        # self.center = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     ConvBn2d(512, 1024,
        #              kernel_size=3, padding=1),
        #     self.activation,
        #     ConvBn2d(1024, 1024,
        #              kernel_size=3, padding=1),
        #     self.activation,
        # )
        self.center = CenterBlock(512, 1024, pool=True, SE=True)

        self.decoder5 = Decoder_v2(512 + 512, 512, 512, convT_channels=1024, SE=True, activation=self.activation)
        self.decoder4 = Decoder_v2(256 + 256, 256, 256, convT_channels=512, SE=True, activation=self.activation)
        self.decoder3 = Decoder_v2(128 + 128, 128, 128, convT_channels=256, SE=True, activation=self.activation)
        self.decoder2 = Decoder_v2(64 + 64, 64, 64, convT_channels=128, SE=True, activation=self.activation)
        self.decoder1 = Decoder_v2(32 + 64, 64, 32, convT_channels=64, SE=True, activation=self.activation)

        # self.reducer5 = nn.Conv2d(512, 32, kernel_size=1, stride=1)
        # self.reducer4 = nn.Conv2d(256, 32, kernel_size=1, stride=1)
        # self.reducer3 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        # self.reducer2 = nn.Conv2d(64, 32, kernel_size=1, stride=1)

        self.reducer5 = HyperBlock(512, 32, SE=True)
        self.reducer4 = HyperBlock(256, 32, SE=True)
        self.reducer3 = HyperBlock(128, 32, SE=True)
        self.reducer2 = HyperBlock(64, 32, SE=True)

        self.logit = nn.Sequential(
            nn.Conv2d(32 * 5, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x, z=None):
        # batch_size,C,H,W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        c = self.center(e4)  # 4

        d5 = self.decoder5(c, e4)  # 8
        d4 = self.decoder4(d5, e3)  # 16
        d3 = self.decoder3(d4, e2)  # 32
        d2 = self.decoder2(d3, e1)  # 64
        d1 = self.decoder1(d2, x)   # 128

        f = torch.cat([
            d1,
            F.upsample(self.reducer2(d2), scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(self.reducer3(d3), scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(self.reducer4(d4), scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(self.reducer5(d5), scale_factor=16, mode='bilinear', align_corners=False),
        ], 1)
        f = F.dropout2d(f, p=0.20)
        logit = self.logit(f)
        return logit

class UNetResNet34_SE_Hyper_v2(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(UNetResNet34_SE_Hyper_v2, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = ELU_1(inplace=True)

        self.resnet = ResNet.resnet34(pretrained=pretrained, activation=self.activation, SE=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.activation,
        )  # 64

        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.center = CenterBlock(512, 64, pool=False, SE=True)

        self.decoder4 = Decoder_v3(256, 64,  64, convT_ratio=1,  SE=True)
        self.decoder3 = Decoder_v3(128, 64,  64, convT_ratio=1,  SE=True)
        self.decoder2 = Decoder_v3(64,  64,  64, convT_ratio=1,  SE=True)
        self.decoder1 = Decoder_v3(64,  64,  64, convT_ratio=1,  SE=True)

        # self.reducer = ConvBn2d(512, 64, kernel_size=1, padding=0)

        self.logit = nn.Sequential(
            ConvBn2d(64 * 5, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def forward(self, x, z):
        # batch_size,C,H,W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        c = self.center(e4)  # 8

        d4 = self.decoder4(c, e3)  # 16
        d3 = self.decoder3(d4, e2)  # 32
        d2 = self.decoder2(d3, e1)  # 64
        d1 = self.decoder1(d2, x)   # 128

        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2,  mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4,  mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8,  mode='bilinear', align_corners=False),
            F.upsample(c,  scale_factor=16, mode='bilinear', align_corners=False)
            ], 1)
        logit = self.logit(f)
        return logit

class UNetResNet34_SE_Hyper_SPP(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(UNetResNet34_SE_Hyper_SPP, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = ELU_1(inplace=True)

        self.resnet = ResNet.resnet34(pretrained=pretrained, activation=self.activation, SE=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.activation,
        )  # 64

        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.center = CenterBlock(512, 64, pool=False, SE=True)

        self.decoder4 = Decoder_v3(256, 64,  64, convT_ratio=1,  SE=True)
        self.decoder3 = Decoder_v3(128, 64,  64, convT_ratio=1,  SE=True)
        self.decoder2 = Decoder_v3(64,  64,  64, convT_ratio=1,  SE=True)
        self.decoder1 = Decoder_v3(64,  64,  64, convT_ratio=1,  SE=True)

        self.reducer = ConvBn2d(64 * 5, 64, kernel_size=1, padding=0)

        self.ppm = PyramidPoolingModule([(128, 128), (64, 64), (42, 42), (21, 21)], 64 * 5)

        self.logit = nn.Sequential(
            ConvBn2d(64 * 5, 128, kernel_size=3, padding=1),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        #mean = [0.485, 0.456, 0.406]
        #std = [0.229, 0.224, 0.225]
        #x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        #x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        #x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        c = self.center(e4)  # 8

        d4 = self.decoder4(c, e3)  # 16
        d3 = self.decoder3(d4, e2)  # 32
        d2 = self.decoder2(d3, e1)  # 64
        d1 = self.decoder1(d2, x)   # 128

        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2,  mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4,  mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8,  mode='bilinear', align_corners=False),
            F.upsample(c,  scale_factor=16, mode='bilinear', align_corners=False)
            ], 1)
        logit = self.logit(f)
        return logit

class UNetResNet34_SE_Hyper_SPP_v2(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(UNetResNet34_SE_Hyper_SPP_v2, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = ELU_1(inplace=True)

        self.resnet = ResNet.resnet34(pretrained=pretrained, activation=self.activation, SE=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.activation,
        )  # 64

        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.center = CenterBlock(512, 64, pool=False, SE=True)

        self.decoder4 = Decoder_v3(256, 64,  64, convT_ratio=1,  SE=True)
        self.decoder3 = Decoder_v3(128, 64,  64, convT_ratio=1,  SE=True)
        self.decoder2 = Decoder_v3(64,  64,  64, convT_ratio=1,  SE=True)
        self.decoder1 = Decoder_v3(64,  64,  64, convT_ratio=1,  SE=True)

        self.reducer = ConvBn2d(512 * 5, 64, kernel_size=1, padding=0)

        # self.ppm = PyramidPoolingModule([(128, 128), (64, 64), (42, 42), (21, 21)], 64 * 5)
        self.ppm =SPP(in_channel=512)

        self.logit = nn.Sequential(
            ConvBn2d(64 * 5, 128, kernel_size=3, padding=1),
            # nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def forward(self, x, z):
        # batch_size,C,H,W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        p1 = self.ppm(e4)
        p2 = self.reducer(p1)
        # c = self.center(p)  # 8

        d4 = self.decoder4(p2, e3)  # 16
        # d4 = self.decoder4(c, e3)  # 16
        d3 = self.decoder3(d4, e2)  # 32
        d2 = self.decoder2(d3, e1)  # 64
        d1 = self.decoder1(d2, x)   # 128

        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2,  mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4,  mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8,  mode='bilinear', align_corners=False),
            F.upsample(p2,  scale_factor=16, mode='bilinear', align_corners=False)
            # F.upsample(c,  scale_factor=16, mode='bilinear', align_corners=False)
            ], 1)
        logit = self.logit(f)
        return logit

class UNetResNet34_SE_Hyper_FPA(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(UNetResNet34_SE_Hyper_FPA, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = ELU_1(inplace=True)

        self.resnet = ResNet.resnet34(pretrained=pretrained, activation=self.activation, SE=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.activation,
        )  # 64

        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.center = CenterBlock(512, 64, pool=False, SE=True)

        self.decoder4 = Decoder_v3(256, 64, 64, convT_ratio=1, SE=True)
        self.decoder3 = Decoder_v3(128, 64, 64, convT_ratio=1, SE=True)
        self.decoder2 = Decoder_v3(64, 64, 64, convT_ratio=1, SE=True)
        self.decoder1 = Decoder_v3(64, 64, 64, convT_ratio=1, SE=True)

        self.reducer = ConvBn2d(256, 64, kernel_size=1, padding=0)

        self.fpa = FPAModule(in_ch=512, out_ch=256)

        self.logit = nn.Sequential(
            ConvBn2d(64 * 5, 128, kernel_size=3, padding=1),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def forward(self, x, z):
        # batch_size,C,H,W = x.shape
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        # x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        # x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x)  # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2)  # 64

        e1 = self.encoder1(p)  # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        f1 = self.fpa(e4)
        f2 = self.reducer(f1)

        d4 = self.decoder4(f2, e3)  # 16
        d3 = self.decoder3(d4, e2)  # 32
        d2 = self.decoder2(d3, e1)  # 64
        d1 = self.decoder1(d2, x)  # 128

        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(f2, scale_factor=16, mode='bilinear', align_corners=False)
        ], 1)
        logit = self.logit(f)
        return logit

class UNetResNet34_PAN_Hyper(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(UNetResNet34_PAN_Hyper, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = ELU_1(inplace=True)

        self.resnet = ResNet.resnet34(pretrained=pretrained, activation=self.activation, SE=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.activation,
        )  # 64

        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.center = CenterBlock(512, 64, pool=False, SE=True)

        # self.decoder4 = Decoder_v3(256, 64, 64, convT_ratio=1, SE=True)
        # self.decoder3 = Decoder_v3(128, 64, 64, convT_ratio=1, SE=True)
        # self.decoder2 = Decoder_v3(64, 64, 64, convT_ratio=1, SE=True)
        # self.decoder1 = Decoder_v3(64, 64, 64, convT_ratio=1, SE=True)

        self.decoder4 = GAUModule(256, 64)
        self.decoder3 = GAUModule(128, 64)
        self.decoder2 = GAUModule(64, 64)
        self.decoder1 = GAUModule(64, 64)

        self.reducer = ConvBn2d(512, 64, kernel_size=1, padding=0)

        # self.fpa = FPAModule(in_ch=512, out_ch=64)
        self.sfpa = SFPAModule(in_ch=512, out_ch=64)

        self.logit = nn.Sequential(
            ConvBn2d(64 * 5, 128, kernel_size=3, padding=1),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        # x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        # x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x)  # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2)  # 64

        e1 = self.encoder1(p)  # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        f1 = self.sfpa(e4)
        # f1 = self.fpa(e4)
        # f2 = self.reducer(f1)

        d4 = self.decoder4(f1, e3)  # 16
        d3 = self.decoder3(d4, e2)  # 32
        d2 = self.decoder2(d3, e1)  # 64
        d1 = self.decoder1(d2, x)  # 128

        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(f1, scale_factor=16, mode='bilinear', align_corners=False)
        ], 1)
        logit = self.logit(f)
        return logit


class UNetResNet34_PAN_Hyper_attention(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(UNetResNet34_PAN_Hyper_attention, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = ELU_1(inplace=True)

        self.resnet = ResNet.resnet34(pretrained=pretrained, activation=self.activation, SE=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.activation,
        )  # 64

        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.center = CenterBlock(512, 64, pool=False, SE=True)

        # self.decoder4 = Decoder_v3(256, 64, 64, convT_ratio=1, SE=True)
        # self.decoder3 = Decoder_v3(128, 64, 64, convT_ratio=1, SE=True)
        # self.decoder2 = Decoder_v3(64, 64, 64, convT_ratio=1, SE=True)
        # self.decoder1 = Decoder_v3(64, 64, 64, convT_ratio=1, SE=True)

        self.decoder4 = GAUModule(256, 64)
        self.decoder3 = GAUModule(128, 64)
        self.decoder2 = GAUModule(64, 64)
        self.decoder1 = GAUModule(64, 64)

        self.reducer = ConvBn2d(512, 64, kernel_size=1, padding=0)

        self.fpa = FPAModule(in_ch=512, out_ch=64)

        self.logit = nn.Sequential(
            ConvBn2d(64, 128, kernel_size=3, padding=1),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        # x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        # x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x)  # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2)  # 64

        e1 = self.encoder1(p)  # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        f1 = self.fpa(e4)
        # f2 = self.reducer(f1)

        d4 = self.decoder4(f1, e3)  # 16
        d3 = self.decoder3(d4, e2)  # 32
        d2 = self.decoder2(d3, e1)  # 64
        d1 = self.decoder1(d2, x)  # 128

        o1 = d1
        o2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False)
        o3 = F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False)
        o4 = F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False)
        o5 = F.upsample(f1, scale_factor=16, mode='bilinear', align_corners=False)

        batchsize, C, height, width = o1.size()

        proj_query2 = o5.view(batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = o4.view(batchsize, -1, width*height)
        energy2 = torch.bmm(proj_query2, proj_key2)
        attention2 = self.softmax(energy2)

        proj_query1 = o3.view(batchsize, -1, width * height).permute(0, 2, 1)
        proj_key1 = o2.view(batchsize, -1, width*height)
        energy1 = torch.bmm(proj_query1, proj_key1)
        attention1 = self.softmax(energy1)

        proj_value = o1.view(batchsize, -1, width*height)

        out2 = torch.bmm(proj_value, attention2.permute(0, 2, 1))
        out2 = out2.view(batchsize, C, height, width)

        out1 = torch.bmm(proj_value, attention1.permute(0, 2, 1))
        out1 = out1.view(batchsize, C, height, width)

        out = self.gamma2*out2 + self.gamma1*out1 + self.gamma*o1
        # f = torch.cat([
        #     d1,
        #     F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
        #     F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
        #     F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
        #     F.upsample(f1, scale_factor=16, mode='bilinear', align_corners=False)
        # ], 1)
        logit = self.logit(out)
        return logit


class UNetResNet50_SE(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, **kwargs):
        super(UNetResNet50_SE, self).__init__(**kwargs)
        self.resnet = ResNet.resnet50(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64


        self.encoder1 = nn.Sequential(self.resnet.layer1, scSqueezeExcitationGate(256))# 256
        self.encoder2 = nn.Sequential(self.resnet.layer2, scSqueezeExcitationGate(512))# 512
        self.encoder3 = nn.Sequential(self.resnet.layer3, scSqueezeExcitationGate(1024))# 1024
        self.encoder4 = nn.Sequential(self.resnet.layer4, scSqueezeExcitationGate(2048))# 2048

        self.center = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBn2d(2048, 1024,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(1024, 1024,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(512 + 1024*2, 512, 512, convT_channels=1024, SE=True)
        self.decoder4 = Decoder(256 + 512*2, 256, 256, convT_channels=512, SE=True)
        self.decoder3 = Decoder(128 + 256*2, 128, 128, convT_channels=256, SE=True)
        self.decoder2 = Decoder(64 + 128*2, 64, 64, convT_channels=128, SE=True)
        self.decoder1 = Decoder(32 + 64, 64, 32, convT_channels=64, SE=True)

        self.logit = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        f = self.center(e4)  # 4

        f = self.decoder5(f, e4)  # 8
        f = self.decoder4(f, e3)  # 16
        f = self.decoder3(f, e2)  # 32
        f = self.decoder2(f, e1)  # 64
        f = self.decoder1(f, x)

        # f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)
        return logit

##########################################################################

class FPNetResNet34(SegmentationNetwork):
    # PyTorch Feature Pyramid Network model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(FPNetResNet34, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = ELU_1(inplace=True)

        self.resnet = ResNet.resnet34(pretrained=pretrained, activation=self.activation)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.activation,
        )  # 64


        self.encoder1 = self.resnet.layer1 # 64
        self.encoder2 = self.resnet.layer2 # 128
        self.encoder3 = self.resnet.layer3 # 256
        self.encoder4 = self.resnet.layer4 # 512

        self.pyramid_features = ExtractPyramidFeatures([64, 128, 256, 512], 256)
        self.pyramid_prediction = PyramidPrediction(256, 128)

        self.agg_conv = ConvBn2d(4 * 128, 256, kernel_size=3)

        self.decoder = Decoder(256 + 64, 128, 128, SE=False, activation=self.activation)

        self.logit = nn.Sequential(
            ConvBn2d(128, 64, kernel_size=3, padding=1),
            self.activation,
            ConvBn2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        P2, P3, P4, P5 = self.pyramid_features(e1, e2, e3, e4)

        f = torch.cat([
            self.pyramid_prediction(P2),
            self.pyramid_prediction(P3, upsampling=2),
            self.pyramid_prediction(P4, upsampling=4),
            self.pyramid_prediction(P5, upsampling=8),
        ], 1)

        f = self.agg_conv(f)

        f = self.decoder(f, x)
        # f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)
        return logit

class RefineNetResNet34(SegmentationNetwork):
    # PyTorch Feature Refine Network model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(RefineNetResNet34, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = ELU_1(inplace=True)

        self.resnet = ResNet.resnet34(pretrained=pretrained, activation=self.activation)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.activation,
        )  # 64


        self.encoder1 = self.resnet.layer1 # 64
        self.encoder2 = self.resnet.layer2 # 128
        self.encoder3 = self.resnet.layer3 # 256
        self.encoder4 = self.resnet.layer4 # 512

        self.refine4 = RefineNetBlock(512, 512, skip_in=None, pool_mode='avg')
        self.refine3 = RefineNetBlock(256, 256, skip_in=512, pool_mode='avg')
        self.refine2 = RefineNetBlock(128, 128, skip_in=256, pool_mode='avg')
        self.refine1 = RefineNetBlock(64, 64, skip_in=128, pool_mode='avg')
        self.refine0 = RefineNetBlock(64, 32, skip_in=64, pool_mode='avg')

        self.logit = nn.Sequential(
            ResidualConvUnit(32),
            ResidualConvUnit(32),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        r4 = self.refine4(e4)       # 8
        r3 = self.refine3(e3, r4)   # 16
        r2 = self.refine2(e2, r3)   # 32
        r1 = self.refine1(e1, r2)   # 64
        r0 = self.refine0(x, r1)    # 128
        logit = self.logit(r0)

        # logit = F.upsample(logit, scale_factor=2, mode='bilinear', align_corners=True)
        return logit

class RefineNetResNet34_SE(SegmentationNetwork):
    # PyTorch Feature Refine Network model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(RefineNetResNet34_SE, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = ELU_1(inplace=True)

        self.resnet = ResNet.resnet34(pretrained=pretrained, activation=self.activation)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.activation,
        )  # 64


        self.encoder1 = nn.Sequential(self.resnet.layer1, scSqueezeExcitationGate(64))# 64
        self.encoder2 = nn.Sequential(self.resnet.layer2, scSqueezeExcitationGate(128))# 128
        self.encoder3 = nn.Sequential(self.resnet.layer3, scSqueezeExcitationGate(256))# 256
        self.encoder4 = nn.Sequential(self.resnet.layer4, scSqueezeExcitationGate(512))# 512

        self.refine4 = RefineNetBlock(512, 512, skip_in=None, pool_mode='avg', SE=True)
        self.refine3 = RefineNetBlock(256, 256, skip_in=512, pool_mode='avg', SE=True)
        self.refine2 = RefineNetBlock(128, 128, skip_in=256, pool_mode='avg', SE=True)
        self.refine1 = RefineNetBlock(64, 64, skip_in=128, pool_mode='avg', SE=True)
        self.refine0 = RefineNetBlock(64, 32, skip_in=64, pool_mode='avg', SE=True)

        self.logit = nn.Sequential(
            ResidualConvUnit(32),
            ResidualConvUnit(32),
            scSqueezeExcitationGate(32),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        r4 = self.refine4(e4)       # 8
        r3 = self.refine3(e3, r4)   # 16
        r2 = self.refine2(e2, r3)   # 32
        r1 = self.refine1(e1, r2)   # 64
        r0 = self.refine0(x, r1)    # 128
        logit = self.logit(r0)

        # logit = F.upsample(logit, scale_factor=2, mode='bilinear', align_corners=True)
        return logit

##########################################################################
##########################################################################
