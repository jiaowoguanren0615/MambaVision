from models.build_cls_models import MambaVision
from timm.models import register_model
import torch
import torch.nn as nn
from pathlib import Path


class SegMambaVisionBackbone(MambaVision):
    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 resolution=224,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        super(SegMambaVisionBackbone, self).__init__(dim,
                                                     in_dim,
                                                     depths,
                                                     window_size,
                                                     mlp_ratio,
                                                     num_heads, **kwargs)
        self.feature_dims = [dim * 2, dim * 4, dim * 8, dim * 8]
        for i in range(len(depths)):
            self.add_module(f"norm{i}", nn.BatchNorm2d(self.feature_dims[i]))

    def forward(self, x):
        out = []
        x = self.patch_embed(x)
        for i in range(len(self.levels)):
            x = self.levels[i](x)
            norm_layer = getattr(self, f"norm{i}")
            out.append(norm_layer(x))
        return out


@register_model
def SegBackbone_mamba_vision_T(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_T.pth.tar")
    # pretrained_cfg = resolve_pretrained_cfg('mamba_vision_T').to_dict()
    # _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter=None)
    model = SegMambaVisionBackbone(depths=[1, 3, 8, 4],
                                   num_heads=[2, 4, 8, 16],
                                   window_size=[8, 8, 14, 7],
                                   dim=80,
                                   in_dim=32,
                                   mlp_ratio=4,
                                   resolution=224,
                                   drop_path_rate=0.2,
                                   **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_model
def SegBackbone_mamba_vision_T2(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_T2.pth.tar")
    # pretrained_cfg = resolve_pretrained_cfg('mamba_vision_T2').to_dict()
    # _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter=None)
    model = SegMambaVisionBackbone(depths=[1, 3, 11, 4],
                                   num_heads=[2, 4, 8, 16],
                                   window_size=[8, 8, 14, 7],
                                   dim=80,
                                   in_dim=32,
                                   mlp_ratio=4,
                                   resolution=224,
                                   drop_path_rate=0.2,
                                   **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_model
def SegBackbone_mamba_vision_S(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_S.pth.tar")
    # pretrained_cfg = resolve_pretrained_cfg('mamba_vision_S').to_dict()
    # _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter=None)
    model = SegMambaVisionBackbone(depths=[3, 3, 7, 5],
                                   num_heads=[2, 4, 8, 16],
                                   window_size=[8, 8, 14, 7],
                                   dim=96,
                                   in_dim=64,
                                   mlp_ratio=4,
                                   resolution=224,
                                   drop_path_rate=0.2,
                                   **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_model
def SegBackbone_mamba_vision_B(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_B.pth.tar")
    # pretrained_cfg = resolve_pretrained_cfg('mamba_vision_B').to_dict()
    # _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter=None)
    model = SegMambaVisionBackbone(depths=[3, 3, 10, 5],
                                   num_heads=[2, 4, 8, 16],
                                   window_size=[8, 8, 14, 7],
                                   dim=128,
                                   in_dim=64,
                                   mlp_ratio=4,
                                   resolution=224,
                                   drop_path_rate=0.3,
                                   layer_scale=1e-5,
                                   layer_scale_conv=None,
                                   **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_model
def SegBackbone_mamba_vision_L(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L.pth.tar")
    # pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L').to_dict()
    # _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter=None)
    model = SegMambaVisionBackbone(depths=[3, 3, 10, 5],
                                   num_heads=[4, 8, 16, 32],
                                   window_size=[8, 8, 14, 7],
                                   dim=196,
                                   in_dim=64,
                                   mlp_ratio=4,
                                   resolution=224,
                                   drop_path_rate=0.3,
                                   layer_scale=1e-5,
                                   layer_scale_conv=None,
                                   **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_model
def SegBackbone_mamba_vision_L2(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L2.pth.tar")
    # pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L2').to_dict()
    # _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter=None)
    model = SegMambaVisionBackbone(depths=[3, 3, 12, 5],
                                   num_heads=[4, 8, 16, 32],
                                   window_size=[8, 8, 14, 7],
                                   dim=196,
                                   in_dim=64,
                                   mlp_ratio=4,
                                   resolution=224,
                                   drop_path_rate=0.3,
                                   layer_scale=1e-5,
                                   layer_scale_conv=None,
                                   **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model
