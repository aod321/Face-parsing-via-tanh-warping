import torchvision.models as models
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops
from torchvision.models import resnet


class NewBackBone_FPN(BackboneWithFPN):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        super(NewBackBone_FPN, self).__init__(backbone, return_layers, in_channels_list, out_channels)

    def forward(self, x):
        c = self.body(x)
        p = self.fpn(c)
        return c, p


def resnet_fpn_backbone(backbone_name, pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': '2', 'layer2': '3', 'layer3': '4', 'layer4': '5'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return NewBackBone_FPN(backbone, return_layers, in_channels_list, out_channels)