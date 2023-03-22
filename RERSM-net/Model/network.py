from __future__ import print_function, division
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore")


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
                          nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                          nn.BatchNorm2d(out_ch),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                          nn.BatchNorm2d(out_ch),
                          nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv_block(x)
        return x


class SkipConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SkipConv, self).__init__()
        self.skip_conv = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                        nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.skip_conv(x)
        return x


class ResEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResEncoder, self).__init__()
        n = 16
        filters = [n, n*2, n*4, n*8, n*16, n*32, n*64]

        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Encoder1 = ConvBlock(in_channel, filters[0])
        self.Encoder2 = ConvBlock(filters[0], filters[1])
        self.Encoder3 = ConvBlock(filters[1], filters[2])
        self.Encoder4 = ConvBlock(filters[2], filters[3])
        self.Encoder5 = ConvBlock(filters[3], filters[4])
        self.Encoder6 = ConvBlock(filters[4], filters[5])
        self.Encoder7 = ConvBlock(filters[5], filters[6])

        self.SkipConv1 = SkipConv(in_channel, filters[0])
        self.SkipConv2 = SkipConv(filters[0], filters[1])
        self.SkipConv3 = SkipConv(filters[1], filters[2])
        self.SkipConv4 = SkipConv(filters[2], filters[3])
        self.SkipConv5 = SkipConv(filters[3], filters[4])
        self.SkipConv6 = SkipConv(filters[4], filters[5])
        self.SkipConv7 = SkipConv(filters[5], filters[6])

        self.Conv_out = nn.Sequential(
                        nn.Conv2d(filters[6], out_channel, kernel_size=1, stride=1, padding=0, bias=True),
                        nn.LeakyReLU(inplace=True))

    def forward(self, x):
        # ResBlock1
        s1 = self.SkipConv1(x)
        e1 = self.Encoder1(x) + s1
        
        e2 = self.MaxPool1(e1)

        # ResBlock2
        s2 = self.SkipConv2(e2)
        e2 = self.Encoder2(e2) + s2
        
        e3 = self.MaxPool2(e2)

        # ResBlock3
        s3 = self.SkipConv3(e3)
        e3 = self.Encoder3(e3) + s3
        
        e4 = self.MaxPool3(e3)

        # ResBlock4
        s4 = self.SkipConv4(e4)
        e4 = self.Encoder4(e4) + s4
        
        e5 = self.MaxPool4(e4)

        # ResBlock5
        s5 = self.SkipConv5(e5)
        e5 = self.Encoder5(e5) + s5
        
        e6 = self.MaxPool5(e5)

        # ResBlock6
        s6 = self.SkipConv6(e6)
        e6 = self.Encoder6(e6) + s6
        
        e7 = self.MaxPool6(e6)

        # ResBlock7
        s7 = self.SkipConv7(e7)
        e7 = self.Encoder7(e7) + s7
        
        # Output layer
        out = self.Conv_out(e7)

        del x, e1, e2, e3, e4, e5, e6, e7, s1, s2, s3, s4, s5, s6, s7
        return out
