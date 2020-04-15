import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import uuid as uid
from model import Hybird
from torch.utils.data import DataLoader
from torchvision import transforms
from preprocess import ToTensor, PrepareLabels
from dataset import Warped_HelenDataset
from template import TemplateModel
from tensorboardX import SummaryWriter
from prefetch_generator import BackgroundGenerator
from augmentation import Stage2DataAugmentation
from tqdm import tqdm

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
parser.add_argument('--pretrain_path', metavar='DIR', default="exp_stage1/checkpoints8625eff6/best.pth.tar", type=str,
                    help='path to save output')

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
            PrepareLabels((128, 128)),
            ToTensor()
        ]),
    'val':
        transforms.Compose([
            PrepareLabels((128, 128)),
            ToTensor()
        ]),
    'test':
        transforms.Compose([
            PrepareLabels((128, 128)),
            ToTensor()
        ])
}

Dataset = {'val': Warped_HelenDataset(root_dir=root_dir['val'],
                                      mode='val',
                                      transform=transforms_list['val']
                                      )}
if args.datamore:
    # Stage 1 augmentation
    stage1_augmentation = Stage2DataAugmentation(dataset=Warped_HelenDataset,
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
        self.model = Hybird(args).to(self.device)
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

        self.criterion = nn.CrossEntropyLoss()
        self.metric = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.train_loader = dataloader['train']
        self.eval_loader = dataloader['val']
        self.ckpt_dir = "exp_stage2/checkpoints%s" % uuid

        self.display_freq = args.display_freq
        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        self.step += 1
        image, labels = batch['image'].to(self.device), batch['parts_labels']
        outer_label, inner_label = labels['outer'], labels['inner']
        in_pred, out_pred = self.model(image)
        outer_label = outer_label.squeeze(1).to(self.device)
        assert outer_label.shape == (image.shape[0], 128, 128)
        assert out_pred.shape == (image.shape[0], 11, 128, 128)
        loss_out = self.criterion(out_pred, outer_label.long())
        loss_in = []
        for i, x in enumerate(['leye', 'reye', 'nose', 'mouth']):
            loss_in.append(self.criterion(in_pred[x], inner_label[i].squeeze(1).to(self.device).long()))
        loss_in = torch.mean(torch.stack(loss_in))
        return loss_in, loss_out

    def eval_error(self):
        loss_in_list = []
        loss_out_list = []
        for batch in tqdm(self.eval_loader):
            image, labels = batch['image'].to(self.device), batch['parts_labels']
            outer_label, inner_label = labels['outer'], labels['inner']
            outer_label = outer_label.squeeze(1).to(self.device)
            in_pred, out_pred = self.model(image)

            assert outer_label.shape == (image.shape[0], 128, 128)
            assert out_pred.shape == (image.shape[0], 11, 128, 128)

            loss_out_list.append(self.criterion(out_pred, outer_label.long()).item())
            loss_in = []
            for i, x in enumerate(['leye', 'reye', 'nose', 'mouth']):
                loss_in.append(self.criterion(in_pred[x], inner_label[i].squeeze(1).to(self.device).long()).item())
            loss_in = np.mean(loss_in)
            loss_in_list.append(loss_in)
        mean_loss_in_error = np.mean(loss_in_list)
        mean_loss_out_error = np.mean(loss_out_list)
        return mean_loss_in_error, mean_loss_out_error

    def train(self):
        self.model.train()
        self.epoch += 1
        for batch in tqdm(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()
            loss_in, loss_out = self.train_loss(batch)
            loss_in.backward(retain_graph=True)
            loss_out.backward()
            self.optimizer.step()

            if self.step % self.display_freq == 0:
                self.writer.add_scalar('loss_in_train_%s' % uuid, loss_in.item(), self.step)
                self.writer.add_scalar('loss_out_train_%s' % uuid, loss_out.item(), self.step)
                print('epoch {}\tstep {}\tloss_in {:.3}\tloss_out{:.3}'.format(self.epoch, self.step,
                                                                               loss_in.item(), loss_out.item()))
        with torch.cuda.device(args.cuda):
            torch.cuda.empty_cache()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            error_in, error_out = self.eval_error()
        if os.path.exists(self.ckpt_dir) is False:
            os.makedirs(self.ckpt_dir)

        if error_in + error_out < self.best_error:
            self.best_error = error_in + error_out
            self.save_state(os.path.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('eval_error%s' % uuid, error_in + error_out, self.epoch)
        print(
            'epoch {}\t mean_error {:.3}\t best_error {:.3}'.format(self.epoch, error_in + error_out, self.best_error))
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

        state['step'] = self.step
        state['epoch'] = self.epoch
        state['best_error'] = self.best_error
        torch.save(state, fname)
        print('save model at {}'.format(fname))


def start_train():
    train = TrainModel()
    for epoch in range(args.epochs):
        train.train()
        train.scheduler.step(epoch)
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
