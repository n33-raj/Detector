import torch
from model.blocks import ConvBlock

def fuse_conv_bn(conv, bn):
    """Fuse Conv2d and BatchNorm2d into one Conv2d"""
    fused_conv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.size()))

    b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.weight.size(0), device=conv.weight.device)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_conv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fused_conv


class FuseLayer:
    @staticmethod
    def fuse_convblock(conv_block):
        if hasattr(conv_block, 'conv') and hasattr(conv_block, 'bn'):
            conv_block.conv = fuse_conv_bn(conv_block.conv, conv_block.bn)
            conv_block.forward = lambda x: conv_block.act(conv_block.conv(x))
            del conv_block.bn
        return conv_block

    @staticmethod
    def fuse_module(module):
        for name, child in module.named_children():
            if isinstance(child, ConvBlock):
                FuseLayer.fuse_convblock(child)
            else:
                FuseLayer.fuse_module(child)
        return module
