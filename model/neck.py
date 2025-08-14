import torch
from model.blocks import ConvBlock, CSPBlock

class Neck(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.top_to_mid = CSPBlock(width[4] + width[5], width[4], depth[0], add=False)
        self.mid_to_small = CSPBlock(width[3] + width[4], width[3], depth[0], add=False)

        self.downsample_mid = ConvBlock(width[3], width[3], 3, 2)
        self.mid_fuse = CSPBlock(width[3] + width[4], width[4], depth[0], add=False)

        self.downsample_top = ConvBlock(width[4], width[4], 3, 2)
        self.top_fuse = CSPBlock(width[4] + width[5], width[5], depth[0], add=False)

    def forward(self, feats):
        feat_small, feat_mid, feat_large = feats
        merged_mid = self.top_to_mid(torch.cat([self.upsample(feat_large), feat_mid], dim=1))
        merged_small = self.mid_to_small(torch.cat([self.upsample(merged_mid), feat_small], dim=1))
        mid_down = self.mid_fuse(torch.cat([self.downsample_mid(merged_small), merged_mid], dim=1))
        top_down = self.top_fuse(torch.cat([self.downsample_top(mid_down), feat_large], dim=1))
        return merged_small, mid_down, top_down
