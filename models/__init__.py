from .spectral import Spectral
from .supspectral import SupSpectral
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2
from .backbones import resnet18_cifar_variant1_mlp1000_norelu, resnet18_cifar_variant1_mlp_norelu
from .backbones import resnet50_mlp8192_norelu_3layer


def get_backbone(backbone, castrate=True, proj_feat_dim=1000):
    backbone = eval(f"{backbone}(featdim={proj_feat_dim})")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg, args=None):
    if model_cfg.name == 'spectral':
        if "mu" not in model_cfg.__dict__:
            model_cfg.mu = 1.0
        model = Spectral(get_backbone(model_cfg.backbone, args.proj_feat_dim), mu=model_cfg.mu, args=args)
    elif model_cfg.name == 'supspectral':
        if "mu" not in model_cfg.__dict__:
            model_cfg.mu = 1.0
        model = SupSpectral(get_backbone(model_cfg.backbone, args.proj_feat_dim), mu=model_cfg.mu, args=args)
    else:
        raise NotImplementedError
    return model






