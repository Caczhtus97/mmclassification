import torch
from fvcore.nn import FlopCountAnalysis
import math


from DANet import DAModule, ChannelAttentionModule, PositionAttentionModule, ScaledDotProductAttention, SimplifiedScaledDotProductAttention
from CBAM import CBAMBlock, ChannelAttention, SpatialAttention
from CoordAttention import CoordAtt
from DualAttention import DualAttention


def main():
    # Dual Attention
    a1 = DAModule(d_model=512, kernel_size=3, H=7, W=7)

    # CBAM
    a2 = CBAMBlock(channel=512, reduction=16, kernel_size=7)

    # CoordAttention
    a3 = CoordAtt(inp=512, oup=512, reduction=32)

    # Dual Attention
    a4 = DualAttention(in_dim=512)

    # [batch_size, channel, height, width]
    t = (torch.rand(128, 512, 7, 7),)

    flops1 = FlopCountAnalysis(a1, t)
    print(f"Dual Attention FLOPs: {round(flops1.total() / 1e6, 3)} M")

    flops2 = FlopCountAnalysis(a2, t)
    print(f"CBAM FLOPs: {round(flops2.total() / 1e6, 3)} M", )

    flops3 = FlopCountAnalysis(a3, t)
    print(f"CoordAttention FLOPs: {round(flops3.total() / 1e6, 3)} M")

    flops4 = FlopCountAnalysis(a4, t)
    print(f"Dual Attention FLOPs: {round(flops4.total() / 1e6, 3)} M")


if __name__ == '__main__':
    main()

