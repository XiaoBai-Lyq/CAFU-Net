from torch import nn
import torch


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.mean(x,1).unsqueeze(1), torch.max(x,1)[0].unsqueeze(1)), dim=1)

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 3
        self.compress = ZPool()
        self.conv = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
        self.bn = nn.BatchNorm2d(1)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.bn(self.conv(x_compress))
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out

if __name__ == '__main__':
    unet = TripletAttention()
    rgb = torch.randn([1,64, 128,128])
    output = unet(rgb)
    print(output.size())