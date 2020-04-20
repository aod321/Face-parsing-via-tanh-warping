from resfpn import resnet_fpn_backbone
import torch.nn as nn


class Hybird(nn.Module):
    def __init__(self, hparam):
        super(Hybird, self).__init__()
        self.backbone = BackBone()
        self.outer_Seg = outer_Seg(n_class=3)

    def forward(self, x):
        C, P = self.backbone(x)
        """" Outer Pred  """
        outer_pred = self.outer_Seg(P['2'])
        # outer Shape(N, 256, 128, 128)

        return outer_pred

# Ouput Shape(N, n_class, 128, 128)
class outer_Seg(nn.Module):
    def __init__(self, n_class):
        super(outer_Seg, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.classifier = nn.Conv2d(256, n_class, kernel_size=1)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.classifier(x)
        return x


# "C4" or S_R = c['4']
# "P2" or S_M = p['2']
class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        self.fpn = resnet_fpn_backbone(backbone_name='resnet18', pretrained=False)

    def forward(self, x):
        x = self.fpn(x)
        return x
