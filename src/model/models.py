import torch

from src.model.hybrid import get_hybrid
from src.model.mobile_net_v2 import get_mobilenet_v2
from src.model.resnet import get_resnet
from src.model.resnet32 import get_resnet32
from src.model.senet import get_seresnet
from src.model.vit import get_vit


def get_model(model_name: str, nclass: int, **kwargs):
    if model_name.startswith('resnet32'):
        model = get_resnet32(model_name, nclass, **kwargs)
    elif model_name.startswith('resnet'):
        model = get_resnet(model_name, nclass, **kwargs)
    elif model_name.startswith('vit'):
        model = get_vit(model_name, nclass, **kwargs)
    elif model_name.startswith('r50'):
        model = get_hybrid(model_name, nclass, **kwargs)
    elif model_name.startswith('mobilenet_v2'):
        model = get_mobilenet_v2('mobilenet_v2', nclass, **kwargs)
    elif model_name.startswith('seresnet'):
        model = get_seresnet(model_name, nclass, **kwargs)

    return model