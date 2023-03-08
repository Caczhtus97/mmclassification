import torch
from fvcore.nn import FlopCountAnalysis
import math


from DANet import DAModule, ChannelAttentionModule, PositionAttentionModule, ScaledDotProductAttention, SimplifiedScaledDotProductAttention
from CBAM import CBAMBlock, ChannelAttention, SpatialAttention
from CoordAttention import CoordAtt
from DualAttention import DualAttention
from SEAttention import SEAttention


def main():
    # Dual Attention
    a1 = DAModule(d_model=512, kernel_size=3, H=7, W=7)

    # CBAM
    a2 = CBAMBlock(channel=512, reduction=16, kernel_size=7)

    # CoordAttention
    a3 = CoordAtt(inp=512, oup=512, reduction=32)

    # Dual Attention
    a4 = DualAttention(in_dim=512)

    # Self Attention
    a5 = ScaledDotProductAttention(d_model=49, d_k=512, d_v=49, h=1)

    # SE Attention
    a6 = SEAttention(channel=512, reduction=16)

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

    flops5 = FlopCountAnalysis(a5, 3*(t[0].view((128, 512, -1)),))
    print(f"Self Attention FLOPs: {round(flops5.total() / 1e6, 3)} M")

    flops6 = FlopCountAnalysis(a6, t)
    print(f"SE Attention FLOPs: {round(flops6.total() / 1e6, 3)} M")



if __name__ == '__main__':
    main()

