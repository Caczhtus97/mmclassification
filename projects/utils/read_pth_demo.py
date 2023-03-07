import torch

pthfile = r'../../checkpoints/seresnet.pth'  # .pth文件的路径
model = torch.load(pthfile, torch.device('cpu'))  # 设置在cpu环境下查询
print('type:')
print(type(model))  # 查看模型字典长度
print('length:')
print(len(model))
print('key:')

for k in model.keys():  # 查看模型字典里面的key
    print(k)

print()
print('value:')
for k in model:  # 查看模型字典里面的value
    if k not in ('state_dict'):
        continue
    for k2 in model[k]:
        print(k2, model[k][k2].shape)