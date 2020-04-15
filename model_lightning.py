from resfpn import resnet_fpn_backbone
from collections import OrderedDict
from torchvision.ops.roi_align import RoIAlign
from torch.nn import SmoothL1Loss, CrossEntropyLoss
from dataset import new_HelenDataset
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader


class Stage1(pl.LightningModule):
    def __init__(self, hparam):
        super(Stage1, self).__init__()
        self.backbone = BackBone()
        self.bbox_regress = ComponentRegress()
        self.criterion = SmoothL1Loss()
        self.args = hparam['args']
        self.hparam = hparam
        self.optimizer = None
        self.scheduler = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def forward(self, x):
        C, P = self.backbone(x)
        bbox = self.bbox_regress(C['4'])
        return bbox

    def training_step(self, batch, batch_idx):
        image, warp_boxes = batch['image'].to(self.device), batch['warp_boxes'].to(self.device)
        pred = self.model(image)
        assert pred.shape == (pred.shape[0], 4, 4)
        assert pred.shape == warp_boxes.shape
        loss = self.criterion(torch.tanh(pred), warp_boxes)
        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_idx):
        image, warp_boxes = batch['image'].to(self.device), batch['warp_boxes'].to(self.device)
        pred = self.model(image)
        assert pred.shape == (pred.shape[0], 4, 4)
        assert pred.shape == warp_boxes.shape
        loss = self.criterion(torch.tanh(pred), warp_boxes)

        output = OrderedDict({
            'val_loss': loss
        })

        return output

    def validation_epoch_end(self, outputs):
        loss_list = []
        for out in outputs:
            loss_list.append(out['val_loss'])

        mean_error = np.mean(loss_list)
        tqdm_dict = {'val_loss': mean_error}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
        return result

    def configure_optimizers(self):
        if self.args.optim == 0:
            self.optimizer = optim.Adam(self.parameters(), self.args.lr)
        elif self.args.optim == 1:
            self.optimizer = optim.SGD(self.parameters(), self.args.lr,
                                       momentum=self.args.momentum, dampening=self.args.dampening,
                                       weight_decay=self.args.decay, nesterov=False)
        elif self.args.optim == 2:
            self.optimizer = optim.SGD(self.parameters(), self.args.lr,
                                       momentum=self.args.momentu, dampening=self.args.dampening,
                                       weight_decay=self.args.decay, nesterov=True)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        return [self.optimizer], [self.scheduler]

    def prepare_data(self):
        self.train_data = new_HelenDataset(root_dir=self.hparam['root_dir']['train'],
                                           mode='train',
                                           transform=self.hparam['transforms']['train'])

        self.val_data = new_HelenDataset(root_dir=self.hparam['root_dir']['val'],
                                         mode='val',
                                         transform=self.hparam['transforms']['val'])

        self.test_data = new_HelenDataset(root_dir=self.hparam['root_dir']['test'],
                                          mode='test',
                                          transform=self.hparam['transforms']['test'])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers)


class Hybird(pl.LightningModule):
    def __init__(self, hparam):
        self.load_from_checkpoint(hparam['pretrain_path'])
        self.backbone = BackBone()
        self.bbox_regress = ComponentRegress()
        self.roi_align = RoIAlign(output_size=(32, 32), spatial_scale=128. / 512., sampling_ratio=-1)
        self.inner_Module = inner_Module(num=4)
        self.outer_Seg = outer_Seg(n_class=2)
        self.criterion = CrossEntropyLoss()

    def forward(self, x):
        # "C4" or S_R = c[2]
        # "P2" or S_M = p['0']

        C, P = self.backbone(x)
        bbox = self.bbox_regress(C['4'])
        # boxes(Tensor[K, 5] or List[Tensor[L, 4]])
        rois = self.roi_align(input=P['2'], rois=bbox)
        # roi (N, 4, 32, 32)
        inner_pred = self.inner_Seg(rois)
        # inner_out is a OrderedDict
        # leye, reye, nose, mouth
        outer_pred = self.outer_Seg(P['2'])

        return inner_pred, outer_pred

    def training_step(self, batch, batch_idx):
        image, labels = batch['image'], batch['labels']
        pred = self(image)
        loss = self.criterion(pred, labels.long())
        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_idx):
        image, labels = batch['image'], batch['labels']
        pred = self(image)
        loss = self.criterion(pred, labels.long())
        hist = self._fast_histogram(pred.argmax(dim=1, keepdim=False).cpu().numpy(),
                                    labels.long().cpu().numpy(),
                                    2, 2
                                    )
        output = OrderedDict({
            'val_loss': loss,
            'val_hist': hist
        })
        return output

    def validation_epoch_end(self, outputs):
        loss_list = []
        hist_list = []
        for out in outputs:
            loss_list.append(out['val_loss'])
            hist_list.append(out['val_hist'])
        mean_error = np.mean(loss_list)
        hist_sum = np.sum(np.stack(hist_list, axis=0), axis=0)
        A = hist_sum[1, :].sum()
        B = hist_sum[:, 1].sum()
        inter_select = hist_sum[1, 1]
        F1 = 2 * inter_select / (A + B)
        tqdm_dict = {'val_loss': mean_error, 'val_F1': F1}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'avg_val_loss': tqdm_dict["val_loss"], 'val_F1': F1}
        return result

    def test_step(self, batch, batch_idx):
        image, labels = batch['image'], batch['labels']
        pred = self(image)
        loss = self.criterion(pred, labels.long())
        hist = self._fast_histogram(pred.argmax(dim=1, keepdim=False).cpu().numpy(),
                                    labels.long().cpu().numpy(),
                                    2, 2
                                    )
        output = OrderedDict({
            'test_loss': loss,
            'test_hist': hist,
        })
        return output

    def test_end(self, outputs):
        loss_list = []
        hist_list = []
        for out in outputs:
            loss_list.append(out['test_loss'])
            hist_list.append(out['test_hist'])

        mean_error = np.mean(loss_list)
        hist_sum = np.sum(np.stack(hist_list, axis=0), axis=0)
        A = hist_sum[1, :].sum()
        B = hist_sum[:, 1].sum()
        inter_select = hist_sum[1, 1]
        F1 = 2 * inter_select / (A + B)
        tqdm_dict = {'test_loss': mean_error, 'test_F1': F1}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'avg_test_loss': tqdm_dict["test_loss"], 'test_F1': F1}
        return result

    def prepare_data(self):
        self.train_data = new_HelenDataset(self.root_dir, 'train')
        self.val_data = new_HelenDataset(self.root_dir, 'val')
        self.test_data = new_HelenDataset(self.root_dir, 'test')

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.args.size, shuffle=True, num_workers=self.args.workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.args.size, shuffle=True, num_workers=self.args.workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.size, shuffle=True, num_workers=self.args.workers)

    @staticmethod
    def _fast_histogram(a, b, na, nb):
        '''
        fast histogram calculation
        ---
        * a, b: non negative label ids, a.shape == b.shape, a in [0, ... na-1], b in [0, ..., nb-1]
        '''
        assert a.shape == b.shape
        assert np.all((a >= 0) & (a < na) & (b >= 0) & (b < nb))
        # k = (a >= 0) & (a < na) & (b >= 0) & (b < nb)
        hist = np.bincount(
            nb * a.reshape([-1]).astype(int) + b.reshape([-1]).astype(int),
            minlength=na * nb).reshape(na, nb)
        assert np.sum(hist) == a.size
        return hist

# Ouput Shape(N, n_class, 128, 128)
class outer_Seg(pl.LightningModule):
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


# Ouput Shape(N, n_class, 128, 128)
class inner_Seg(pl.LightningModule):
    def __init__(self, n_class):
        super(inner_Seg, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.classifier = nn.Conv2d(256, n_class, kernel_size=1)

    def forward(self, x):
        # x Shape(4 * N, 3, 32, 32)
        x = self.upsample(self.bn1(self.relu(self.conv1(x))))
        x = self.upsample(self.bn2(self.relu(self.conv2(x))))
        x = self.classifier(x)
        return x


class ComponentRegress(pl.LightningModule):
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
class BackBone(pl.LightningModule):
    def __init__(self):
        super(BackBone, self).__init__()
        self.fpn = resnet_fpn_backbone(backbone_name='resnet18', pretrained=False)

    def forward(self, x):
        x = self.fpn(x)
        return x