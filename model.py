from resfpn import resnet_fpn_backbone
from collections import OrderedDict
from torchvision.ops.roi_align import RoIAlign
import torch
import torch.nn as nn


def napply_mat_tensor(coords, matrix):
    #  coords Shape(N, 4, 4)
    # matrix Shape(N, 3, 3)
    temp = coords.view(-1, 8, 2).permute(0, 2, 1)
    x = temp[:, 0]
    y = temp[:, 1]
    src = torch.stack([x, y, torch.ones_like(x)], dim=1)
    results = []
    for i in range(src.shape[0]):
        dst = src[i].T @ matrix[i].T
        dst[dst[:, 2] == 0, 2] = torch.finfo(float).eps
        dst[:, :2] /= dst[:, 2:3]
        results.append(dst[:, :2])
    results = torch.stack(results).view(-1, 4, 4)
    assert results.shape == (src.shape[0], 4, 4)
    return results


class Stage1(nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.backbone = BackBone()
        self.bbox_regress = ComponentRegress()

    def forward(self, x):
        C, P = self.backbone(x)
        bbox = self.bbox_regress(C['4'])
        return bbox


class Hybird(nn.Module):
    def __init__(self, hparam):
        super(Hybird, self).__init__()
        self.backbone = BackBone()
        self.bbox_regress = ComponentRegress()
        self.roi_align = RoIAlign(output_size=(32, 32), spatial_scale=128. / 512., sampling_ratio=-1)
        self.inner_Module = inner_Module(num=4)
        self.outer_Seg = outer_Seg(n_class=11)
        self.load_from_checkpoint(hparam.pretrain_path, torch.device(f"cuda:{hparam.cuda}"))

    def forward(self, x):
        C, P = self.backbone(x)
        bbox = self.bbox_regress(C['4'])
        """"" Mapping """
        bbox = torch.tanh(bbox) * 256. - 256.
        """" Bbox Padding """
        # non-mouth
        bbox[:3] += bbox[:3] * 0.05
        # mouth
        bbox[3] += bbox[3] * 0.1
        """" Roi Align  """
        # boxes Shape(10, 4, 4)
        boxes = [bbox[:, i]
                 for i in range(4)]
        roi_preds = self.roi_align(input=P['2'], rois=boxes)
        # roi_preds Shape(4 * N, 256, 32, 32)
        assert roi_preds.shape == (roi_preds.shape[0], 256, 32, 32)

        """" Inner Pred  """
        inner_pred = self.inner_Module(roi_preds)
        # inner_out is a OrderedDict
        # leye, reye, nose, mouth
        # Each of them has Shape(N, 11, 128, 128)

        """" Outer Pred  """
        outer_pred = self.outer_Seg(P['2'])
        # outer Shape(N, 11, 128, 128)

        return inner_pred, outer_pred

    def load_from_checkpoint(self, path, device):
        state = torch.load(path, map_location=device)
        self.backbone.load_state_dict(state['backbone'])
        print("bacbone loaded")
        self.bbox_regress.load_state_dict(state['regress'])
        print("bbox_regress loaded")


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


class inner_Module(nn.Module):
    def __init__(self, num):
        super(inner_Module, self).__init__()
        self.segnets = nn.ModuleList([inner_Seg(11) for _ in range(num)])
        self.names = ['leye', 'reye', 'nose', 'mouth']
        self.num = num

    def forward(self, x):
        # Input x Shape(4 * N, 256, 32, 32)
        x = x.view(self.num, -1, 256, 32, 32)
        # Shape(4, N, 256, 32, 32)
        outputs = OrderedDict()
        for i in range(self.num):
            outputs[self.names[i]] = self.segnets[i](x[i])
        return outputs


# Ouput Shape(N, n_class, 128, 128)
class inner_Seg(nn.Module):
    def __init__(self, n_class):
        super(inner_Seg, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.classifier = nn.Conv2d(256, n_class, kernel_size=1)

    def forward(self, x):
        # x Shape(4 * N, 3, 32, 32)
        x = self.upsample(self.bn1(self.relu(self.conv1(x))))
        x = self.upsample(self.bn2(self.relu(self.conv2(x))))
        x = self.classifier(x)
        # x Shape(4*N, 11, 128, 128)
        return x


class ComponentRegress(nn.Module):
    def __init__(self):
        super(ComponentRegress, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(256, 320, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(320)
        self.conv2 = nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(320)
        self.conv3 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(1280)
        self.avpool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Linear(1280, 16)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.relu(self.avpool(x))
        x = self.fc(x.view(-1, 1280))
        return x.view(-1, 4, 4)


# "C4" or S_R = c['4']
# "P2" or S_M = p['2']
class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        self.fpn = resnet_fpn_backbone(backbone_name='resnet18', pretrained=False)

    def forward(self, x):
        x = self.fpn(x)
        return x
