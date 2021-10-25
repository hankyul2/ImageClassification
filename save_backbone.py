from collections import OrderedDict

import torch


ckpt_path = 'log/resnet32_110_cifar100/IM-126/epoch=53_step=0.00_loss=4.070.ckpt'
state_dict_path = 'pretrained/small_sample/cifar100_small_sample_20_resnet32_110.pth'
state_dict = OrderedDict()
for k, v in torch.load(ckpt_path, map_location='cpu')['state_dict'].items():
    print(k)
    if 'backbone' in k:
        state_dict['.'.join(k.split('.')[1:])] = v
    if 'fc' in k:
        state_dict[k] = v

with open(state_dict_path, 'wb') as f:
    torch.save(state_dict,f)