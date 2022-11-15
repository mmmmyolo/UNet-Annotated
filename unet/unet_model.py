""" Full assembly of the parts to form the complete network """

#导入unet_parts文件中所有方法
from .unet_parts import *


class UNet(nn.Module):
    #初始化函数，对UNet类进行初始化
    def __init__(self, n_channels, n_classes, bilinear=False):
        #调用父类的初始化函数，对nn.Module类进行初始化
        super(UNet, self).__init__()
        #输入图片的通道数
        self.n_channels = n_channels
        #每个像素点的概率类别，例如只有背景和一个类，n_classes设置为1
        self.n_classes = n_classes
        #是否使用双线性插值
        self.bilinear = bilinear

        #完成unet网络左边的部分的一层中的两个卷积
        self.inc = DoubleConv(n_channels, 64)
        #第一次下采样，通道数变成128
        self.down1 = Down(64, 128)
        #第二次下采样，通道数变成256
        self.down2 = Down(128, 256)
        #第三次下采样，通道数变成512
        self.down3 = Down(256, 512)
        #如果时双线性插值则为2否则为1
        factor = 2 if bilinear else 1
        #第四次下采样，通道数变成1024/factor
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
