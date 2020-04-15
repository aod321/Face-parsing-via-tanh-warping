import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import uuid as uid
from model import Stage1
from torch.utils.data import DataLoader
from torchvision import transforms
from preprocess import ToTensor
from dataset import Warped_HelenDataset
from template import TemplateModel
from tensorboardX import SummaryWriter
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from augmentation import DataAugmentation
import pytorch_lightning as pl

uuid = str(uid.uuid1())[0:8]
print(uuid)

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=9, type=int, help="Which GPU to train.")
parser.add_argument("--optim", default=0, type=int, help="Optimizer: 0: Adam, 1: SGD, 2:SGD with Nesterov")
parser.add_argument("--datamore", default=1, type=int, help="Data Augmentation")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size to use during training.")
parser.add_argument("--workers", default=10, type=int, help="Workers")
parser.add_argument("--display_freq", default=8, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
parser.add_argument("--decay", default=0.01, type=float, help="Weight decay")
parser.add_argument("--dampening", default=0, type=float, help="dampening for momentum")

args = parser.parse_args()
print(args)
# Dataset and Dataloader
# Dataset Read_in Part
root_dir = {
    'train': "/home/yinzi/data4/warped/train",
    'val': "/home/yinzi/data4/warped/val",
    'test': "/home/yinzi/data4/warped/test"
}


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


transforms_list = {
    'train':
        transforms.Compose([
            ToTensor()
        ]),
    'val':
        transforms.Compose([
            ToTensor()
        ]),
    'test':
        transforms.Compose([
            ToTensor()
        ])
}

Dataset = {'val': Warped_HelenDataset(root_dir=root_dir['val'],
                                      mode='val',
                                      transform=transforms_list['val']
                                      )}

if args.datamore:
    # Stage 1 augmentation
    stage1_augmentation = DataAugmentation(dataset=Warped_HelenDataset,
                                           root_dir=root_dir
                                           )
    train_dataset = stage1_augmentation.get_dataset()
else:
    train_dataset = {'train': Warped_HelenDataset(root_dir=root_dir['train'],
                                                  mode='train',
                                                  transform=transforms_list['train']
                                                  )}
Dataset.update(train_dataset)
dataloader = {x: DataLoaderX(Dataset[x], batch_size=args.batch_size,
                             shuffle=True, num_workers=args.workers)
              for x in ['train', 'val']
              }


class TrainModel(TemplateModel):

    def __init__(self):
        super(TrainModel, self).__init__()
        self.args = args

        self.writer = SummaryWriter('log')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.model = Stage1().to(self.device)
        if args.optim == 0:
            self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        elif args.optim == 1:
            self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,
                                       momentum=self.args.momentum, dampening=self.args.dampening,
                                       weight_decay=self.args.decay, nesterov=False)
        elif args.optim == 2:
            self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,
                                       momentum=self.args.momentu, dampening=self.args.dampening,
                                       weight_decay=self.args.decay, nesterov=True)

        self.criterion = nn.SmoothL1Loss()
        self.metric = nn.SmoothL1Loss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.train_loader = dataloader['train']
        self.eval_loader = dataloader['val']
        self.ckpt_dir = "exp_stage1/checkpoints%s" % uuid

        self.display_freq = args.display_freq
        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        self.step += 1
        image, warp_boxes = batch['image'].to(self.device), batch['warp_boxes'].to(self.device)
        pred = self.model(image)
        assert pred.shape == (pred.shape[0], 4, 4)
        assert pred.shape == warp_boxes.shape
        loss = self.criterion(torch.tanh(pred), warp_boxes)
        return loss

    def eval_error(self):
        loss_list = []
        for batch in tqdm(self.eval_loader):
            image, warp_boxes = batch['image'].to(self.device), batch['warp_boxes'].to(self.device)
            pred = self.model(image)
            assert pred.shape == (pred.shape[0], 4, 4)
            assert pred.shape == warp_boxes.shape
            loss_list.append(self.criterion(torch.tanh(pred), warp_boxes).item())
        mean_error = np.mean(loss_list)
        return mean_error, None

    def train(self):
        self.model.train()
        self.epoch += 1
        for batch in tqdm(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()
            loss = self.train_loss(batch)
            loss.backward()
            self.optimizer.step()

            if self.step % self.display_freq == 0:
                self.writer.add_scalar('loss_train_%s' % uuid, loss.item(), self.step)
                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, loss.item()))
        with torch.cuda.device(args.cuda):
            torch.cuda.empty_cache()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            error, _ = self.eval_error()
        if os.path.exists(self.ckpt_dir) is False:
            os.makedirs(self.ckpt_dir)

        if error < self.best_error:
            self.best_error = error
            self.save_state(os.path.join(self.ckpt_dir, 'best.pth.tar'), False)

        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('eval_error%s' % uuid, error, self.epoch)
        print('epoch {}\t mean_error {:.3}\t best_error {:.3}'.format(self.epoch, error, self.best_error))
        with torch.cuda.device(args.cuda):
            torch.cuda.empty_cache()

    def save_state(self, fname, optim=True):
        state = {}
        if isinstance(self.model, torch.nn.DataParallel):
            state['model'] = self.model.module.state_dict()
        else:
            state['model'] = self.model.state_dict()
        if optim:
            state['optimizer'] = self.optimizer.state_dict()

        state['backbone'] = self.model.backbone.state_dict()
        state['regress'] = self.model.bbox_regress.state_dict()
        state['step'] = self.step
        state['epoch'] = self.epoch
        state['best_error'] = self.best_error
        torch.save(state, fname)
        print('save model at {}'.format(fname))

    def paste_back(self):
        pass


def start_train():
    train = TrainModel()
    for epoch in range(args.epochs):
        train.train()
        train.scheduler.step(epoch)
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
