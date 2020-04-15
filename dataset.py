from torch.utils.data import Dataset
import os
from glob import glob
import re
from PIL import Image
import cv2
import torchvision.transforms.functional as TF
import pickle
import numpy as np


class new_HelenDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = glob(os.path.join(root_dir, '*_image.jpg'))
        length = len(self.image_dir)
        self.mode = mode
        self.name_list = sorted([re.compile('.*/(.*)_image.jpg', re.S).findall(self.image_dir[i])[0]
                                 for i in range(length)])
        if self.mode == 'train':
            self.name_list = self.name_list[:length - 230]
        if self.mode == 'val':
            self.name_list = self.name_list[length - 230:]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        image_path = os.path.join(self.root_dir, name + "_image.jpg")
        labels_path = os.path.join(self.root_dir, name + '_label.png')
        image = Image.open(image_path)
        labels = TF.to_pil_image(cv2.imread(labels_path, cv2.IMREAD_GRAYSCALE))

        sample = {'image': image,
                  'labels': labels,
                  'name': name
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample


class Warped_HelenDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = glob(os.path.join(root_dir, '*_image.png'))
        length = len(self.image_dir)
        self.mode = mode
        self.name_list = sorted([re.compile('.*/(.*)_image.png', re.S).findall(self.image_dir[i])[0]
                                 for i in range(length)])
        self.root_dir = root_dir
        self.transform = transform
        with open(os.path.join(root_dir, f'{mode}.p'), 'rb') as fp:
            self.others = pickle.load(fp)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        image_path = os.path.join(self.root_dir, name + "_image.png")
        labels_path = os.path.join(self.root_dir, name + '_label.png')
        image = Image.open(image_path)
        labels = TF.to_pil_image(cv2.imread(labels_path, cv2.IMREAD_GRAYSCALE))
        orig_size, boxes, warp_boxes, params = self.others[name]

        sample = {'image': image,
                  'labels': labels,
                  'name': name,
                  'orig_size': orig_size,
                  'boxes': boxes.squeeze(0),
                  'warp_boxes': warp_boxes.squeeze(0),
                  'params': TF.to_tensor(np.array(params, dtype=np.float32)).view(3, 3),
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample
