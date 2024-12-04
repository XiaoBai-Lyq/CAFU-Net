from thop import profile
from torch import nn
import torch
from liuzhouwork_UNet.model import Triplet_Attention
#=================================

class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 5, padding=2)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn + x

class GateFusion(nn.Module):
    def __init__(self, in_planes):
        self.init__ = super(GateFusion, self).__init__()

        self.gate_1 = nn.Conv2d(in_planes * 2, 1, kernel_size=1)
        self.gate_2 = nn.Conv2d(in_planes * 2, 1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        ###
        cat_fea = torch.cat([x1, x2], dim=1)

        ###
        att_vec_1 = self.bn1(self.gate_1(cat_fea))
        att_vec_2 = self.bn2(self.gate_2(cat_fea))

        att_vec_cat = torch.cat([att_vec_1, att_vec_2], dim=1)
        att_vec_soft = self.softmax(att_vec_cat)

        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2

        return x_fusion

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512,2,2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256,2,2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128,2,2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64,2,2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.dropout = nn.Dropout2d(p=0.5)
        self.softmax = nn.Softmax()

        self.trp1 = Triplet_Attention.TripletAttention()

        self.up_cross1 = nn.ConvTranspose2d(128, 64,2,2)
        self.up_cross2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_cross3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_cross4 = nn.ConvTranspose2d(1024, 512, 2, 2)

        self.gate1 = GateFusion(64)
        self.gate2 = GateFusion(128)
        self.gate3 = GateFusion(256)
        self.gate4 = GateFusion(512)

        self.lsk1 = LSKblock(64)
        self.lsk2 = LSKblock(128)
        self.lsk3 = LSKblock(256)
        self.lsk4 = LSKblock(512)

    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(self.lsk1(c1))
        c2 = self.conv2(p1)

        up_cross1 = self.up_cross1(c2)
        cat1 = self.gate1(c1, up_cross1)
        trp1 = self.trp1(cat1)


        p2 = self.pool2(self.lsk2(c2))
        c3 = self.conv3(p2)

        up_cross2 = self.up_cross2(c3)
        cat2 = self.gate2(c2, up_cross2)
        trp2 = self.trp1(cat2)

        p3 = self.pool3(self.lsk3(c3))
        c4 = self.conv4(p3)

        up_cross3 = self.up_cross3(c4)
        cat3 = self.gate3(c3, up_cross3)
        trp3 = self.trp1(cat3)


        p4 = self.pool4(self.lsk4(c4))
        c5 = self.conv5(p4)

        up_cross4 = self.up_cross4(c5)
        cat4 = self.gate4(c4,up_cross4)
        trp4 = self.trp1(cat4)


        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, trp4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, trp3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, trp2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, trp1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        return c10

#==================================================================================
if __name__ == '__main__':
    unet = Unet(1,1)
    rgb = torch.randn([1, 1, 128, 128])
    out1 = unet(rgb).size()

    flops, params = profile(unet, inputs=(rgb,))
    flop_g = flops / (10 ** 9)
    param_mb = params * 4 / (1024 * 1024)  # 转换为MB

    print(f"模型的FLOP数量：{flop_g}G")
    print(f"参数数量: {param_mb} MB")
    print(out1)



