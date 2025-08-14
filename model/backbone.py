import torch
from model.blocks import ConvBlock, CSPBlock, SPPBlock

class Backbone(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.stage1 = torch.nn.Sequential(ConvBlock(width[0], width[1], 3, 2))
        self.stage2 = torch.nn.Sequential(
            ConvBlock(width[1], width[2], 3, 2),
            CSPBlock(width[2], width[2], depth[0])
        )
        self.stage3 = torch.nn.Sequential(
            ConvBlock(width[2], width[3], 3, 2),
            CSPBlock(width[3], width[3], depth[1])
        )
        self.stage4 = torch.nn.Sequential(
            ConvBlock(width[3], width[4], 3, 2),
            CSPBlock(width[4], width[4], depth[2])
        )
        self.stage5 = torch.nn.Sequential(
            ConvBlock(width[4], width[5], 3, 2),
            CSPBlock(width[5], width[5], depth[0]),
            SPPBlock(width[5], width[5])
        )

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return s3, s4, s5
