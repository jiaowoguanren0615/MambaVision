from models.segformer_head import SegFormerHead
from models.upernet_head import UPerHead
import torch.nn.functional as F
from models.seg_model_backbones import *

class SegMambaVisionModel(nn.Module):
    def __init__(self, backbone, num_classes=19, use_segformer_head=False, **kwargs):
        super(SegMambaVisionModel, self).__init__()

        self.backbone = eval(backbone + '()')

        if use_segformer_head == True:
            self.decode_head = SegFormerHead(self.backbone.feature_dims, 256 if 'T' in backbone or 'S' in backbone else 768,
                                             num_classes)
        else:
            self.decode_head = UPerHead(self.backbone.feature_dims, 128 if 'T' in backbone or 'S' in backbone else 768,
                                        num_classes)

    def forward(self, x):
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y


@register_model
def SegMambaVision_T(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    backbone = 'SegBackbone_mamba_vision_T'
    model = SegMambaVisionModel(backbone, **kwargs)
    return model


@register_model
def SegMambaVision_T2(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    backbone = 'SegBackbone_mamba_vision_T2'
    model = SegMambaVisionModel(backbone, **kwargs)
    return model


@register_model
def SegMambaVision_S(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    backbone = 'SegBackbone_mamba_vision_S'
    model = SegMambaVisionModel(backbone, **kwargs)
    return model


@register_model
def SegMambaVision_B(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    backbone = 'SegBackbone_mamba_vision_B'
    model = SegMambaVisionModel(backbone, **kwargs)
    return model


@register_model
def SegMambaVision_L(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    backbone = 'SegBackbone_mamba_vision_L'
    model = SegMambaVisionModel(backbone, **kwargs)
    return model


@register_model
def SegMambaVision_L2(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    backbone = 'SegBackbone_mamba_vision_L2'
    model = SegMambaVisionModel(backbone, **kwargs)
    return model


# if __name__ == '__main__':
#     import torch
#     input_data = torch.randn(2, 3, 224, 224).cuda()
#     model = SegMambaVision_T().cuda()
#     y = model(input_data)
#     print(y.shape)