import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import uuid as uid
from model import test_Stage1
from torch.utils.data import DataLoader
from torchvision import transforms
from preprocess import Warping
from dataset import new_HelenDataset
import torchvision.transforms.functional as TF
from tqdm import tqdm
from PIL import Image
import pickle
from template import TemplateModel
from tensorboardX import SummaryWriter
from prefetch_generator import BackgroundGenerator
from multiprocessing import set_start_method
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass

preprocess_device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# preprocess_device = torch.device("cpu")

# Dataset and Dataloader
# Dataset Read_in Part
root_dir = {
    'train': "/home/yinzi/data3/relabel_helen/helenstar_release/train",
    'val': "/home/yinzi/data3/relabel_helen/helenstar_release/train",
    'test': "/home/yinzi/data3/relabel_helen/helenstar_release/test"
}

transforms_list = {
    'train':
        transforms.Compose([
            Warping((512, 512), device=preprocess_device)
        ]),
    'val':
        transforms.Compose([
            Warping((512, 512), device=preprocess_device)
        ]),
    'test':
        transforms.Compose([
            Warping((512, 512), device=preprocess_device)
        ])
}

Dataset = {x: new_HelenDataset(root_dir=root_dir[x],
                               mode=x,
                               transform=transforms_list[x]
                               )
           for x in ['train', 'test', 'val']
           }

dataloader = {x: DataLoader(Dataset[x], batch_size=1,
                            shuffle=True, num_workers=0)
              for x in ['train', 'test', 'val']
              }
outpath = "/home/yinzi/data4/warped/"

if __name__ == '__main__':
    for x in ['train', 'test', 'val']:
        saveDict = {}
        for batch in tqdm(dataloader[x]):
            name = batch['name']
            image = batch['image']
            labels = batch['labels']

            saveDict.update({name[0]: [batch['orig_size'], batch['boxes'], batch['warp_boxes'], batch['params']]
                             })

            full_path = os.path.join(outpath, x)
            os.makedirs(full_path, exist_ok=True)

            TF.to_pil_image(image[0]).save(os.path.join(full_path, name[0] + '_image.png'),
                                           format='PNG', compress_level=0)

            out_label = TF.to_pil_image(labels[0][0].numpy().astype(np.uint8), mode='L')
            out_label.save(os.path.join(full_path, name[0] + '_label.png'),
                           format='PNG', compress_level=0)

        with open(os.path.join(full_path, f'{x}.p'), 'wb') as fp:
            pickle.dump(saveDict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"{x} Done!")

    print("All Done!")
