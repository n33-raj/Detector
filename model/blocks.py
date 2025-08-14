import torch

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

class ConvBlock(torch.nn.Module):
    """Conv -> BN -> SiLU"""
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.act = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_residual = add
        self.blocks = torch.nn.Sequential(
            ConvBlock(ch, ch, 3),
            ConvBlock(ch, ch, 3)
        )

    def forward(self, x):
        return self.blocks(x) + x if self.add_residual else self.blocks(x)


class CSPBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.split_conv = ConvBlock(in_ch, out_ch // 2)
        self.path_conv = ConvBlock(in_ch, out_ch // 2)
        self.final_conv = ConvBlock((2 + n) * out_ch // 2, out_ch)
        self.residual_blocks = torch.nn.ModuleList([ResidualBlock(out_ch // 2, add) for _ in range(n)])

    def forward(self, x):
        left = self.split_conv(x)
        right = self.path_conv(x)
        out_list = [left, right]
        out_list.extend(block(out_list[-1]) for block in self.residual_blocks)
        return self.final_conv(torch.cat(out_list, dim=1))


class SPPBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.reduce_conv = ConvBlock(in_ch, in_ch // 2)
        self.final_conv = ConvBlock(in_ch * 2, out_ch)
        self.maxpool = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.reduce_conv(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.final_conv(torch.cat([x, y1, y2, y3], 1))
