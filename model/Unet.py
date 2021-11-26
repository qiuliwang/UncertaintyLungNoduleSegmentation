import torch.nn as nn
import torch
from math import sqrt
from ._utils import *

def conv_relu(in_channel,out_channel, kernel, stride=1, padding=0):
    """
    
    conv-bn-relu for unet
    :param in_channel: 输入通道
    :param out_channel: 输出通道
    :param kernel: 卷积和
    :param stride: 步长
    :param padding:
    :return:
    """
    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        nn.BatchNorm2d(out_channel, eps=1e-3),
        nn.ReLU(True)
    )
    return layer

class DoubleConv(nn.Module):
    """
    Double Conv for U-Net
    """
    def __init__(self, in_ch, out_ch, k_1=3, k_2=3):
        super(DoubleConv, self).__init__()
        padding_1 = cal_same_padding(k_1)
        padding_2 = cal_same_padding(k_2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k_1, padding=padding_1),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # Mish(),
            nn.Conv2d(out_ch, out_ch, k_2, padding=padding_2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        return self.conv(x)

class Inception(nn.Module):
    """
    一个即插即用的3路inceptionm模块
    输入尺寸和输出尺寸一致
    Example:
    >>> Inception(64,64//4,x,64//4,x,64//4,64//4)
    """
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        """
        当 in_channel =  out1_1 + out2_3 + out3_5 + out4_1时
        输入和输出尺寸一致
        """
        super(Inception, self).__init__()

        # 定义inception模块第一条线路
        self.branch1x1 = conv_relu(in_channel, out1_1, 1)

        # 定义inception模块第二条线路
        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )

        # 定义inception模块的第三条线路
        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )

        # 定义inception模块第四条线路
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)
        )

    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)

        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Unet(nn.Module):
    def __init__(self, in_dim, out_dim):

        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_dim, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res1 = Inception(64,16,48,16,64,16,16)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)


        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_dim, 1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, inputs):
        c1 = self.conv1(inputs)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)  
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)  # 256 *48
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)       
        c10 = self.conv10(c9)


        return c10, c9


        


