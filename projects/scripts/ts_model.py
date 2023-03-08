import torch
from fvcore.nn import FlopCountAnalysis

from projects.models import SEDAResNet
from mmcls.models import SEResNet


def calc_FLOPs(model, inp, name):
    flops = FlopCountAnalysis(model, inp)
    print(f"{name} FLOPs: {round(flops.total() / 1e6, 3)} M")


def compare_model_FLOPs():
    inp = torch.rand(1, 3, 224, 224)
    se_layer = SEResNet(depth=50)
    se_da_layer = SEDAResNet(depth=50)

    calc_FLOPs(se_layer, inp, 'SEResNet')
    calc_FLOPs(se_da_layer, inp, 'SEDAResNet')

def main():
    layer = SEDAResNet(depth=50)
    layer.eval()
    inp = torch.rand(1, 3, 224, 224)
    outs = layer.forward(inp)
    for out in outs:
        print(tuple(out.shape))

    compare_model_FLOPs()


if __name__ == '__main__':
    main()