from collections import OrderedDict

import torch


ckpt_path = 'log/resnet32_110_cifar100/IM-113/epoch=56_step=0.00_loss=3.665.ckpt'
state_dict_path = 'pretrained/noisy/resnet32_110_cifar100/noisy_ratio_3.pth'
state_dict = OrderedDict()
for k, v in torch.load(ckpt_path, map_location='cpu')['state_dict'].items():
    print(k)
    if 'backbone' in k:
        state_dict['.'.join(k.split('.')[1:])] = v
    if 'fc' in k:
        state_dict[k] = v

with open(state_dict_path, 'wb') as f:
    torch.save(state_dict,f)