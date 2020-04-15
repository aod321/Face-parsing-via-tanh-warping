from torchvision.transforms import transforms
from model import Hybird
import pytorch_lightning as pl
import argparse
from collections import OrderedDict
from preprocess import Warping

hparam = OrderedDict()
parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=list, default=[5],
                    help='Select gpus')
parser.add_argument('--save-path', metavar='DIR', default="checkpoints", type=str,
                    help='path to save output')
parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                    help='supports three options dp, ddp, ddp2')
parser.add_argument("--optim", default=0, type=int, help="Optimizer: 0: Adam, 1: SGD, 2:SGD with Nesterov")
parser.add_argument('--precision', default=32, type=int,
                    help='Use 32bit or 16 bit precision')
parser.add_argument('--benchmark', dest='benchmark', action='store_true',
                    help='Cudann benchmark')
parser.add_argument('--seed', type=int, default=42,
                    help='seed for initializing training. ')
parser.add_argument('--amp_level', type=str, default='O1', choices=('O0', 'O1', 'O2', 'O3'),
                    help='amp_level')
parser.add_argument("--accumulate", default=2, type=int, help="Accumulate_grad_batches")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size to use during training.")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
parser.add_argument("--decay", default=0.01, type=float, help="Weight decay")
parser.add_argument("--dampening", default=0, type=float, help="dampening for momentum")

root_dir = {
    'train': "/home/yinzi/data3/relabel_helen/helenstar_release/train",
    'val': "/home/yinzi/data3/relabel_helen/helenstar_release/train",
    'test': "/home/yinzi/data3/relabel_helen/helenstar_release/test"
}

transforms_list = {
    'train':
        transforms.Compose([
            Warping((512, 512))
        ]),
    'val':
        transforms.Compose([
            Warping((512, 512))
        ]),
    'test':
        transforms.Compose([
            Warping((512, 512))
        ])
}

pretrain_path = ""

hparam['args'] = parser.parse_args()
hparam['root_dir'] = root_dir
hparam['transforms'] = transforms_list
hparam['pretrain_path'] = pretrain_path

print(hparam)

root_dir = {
    'train': "/home/yinzi/data3/relabel_helen/helenstar_release/train",
    'val': "/home/yinzi/data3/relabel_helen/helenstar_release/train",
    'test': "/home/yinzi/data3/relabel_helen/helenstar_release/test"
}

transforms_list = {
    'train':
        transforms.Compose([
            Warping((512, 512))
        ]),
    'val':
        transforms.Compose([
            Warping((512, 512))
        ]),
    'test':
        transforms.Compose([
            Warping((512, 512))
        ])
}


model = Hybird(hparam)
trainer = pl.Trainer(
    gpus=4,
)
trainer.test(model)
