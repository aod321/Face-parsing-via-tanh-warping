from torch.utils.data import ConcatDataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from preprocess import ToTensor, GaussianNoise, RandomAffine


class DataAugmentation(object):
    def __init__(self, dataset, root_dir):
        self.augmentation_name = ['origin', 'choice1', 'choice2', 'choice3', 'choice4']
        self.randomchoice = None
        self.transforms = None
        self.transforms_list = None
        self.dataset = dataset
        self.root_dir = root_dir
        self.set_choice()
        self.set_transformers()
        self.set_transforms_list()

    def set_choice(self):
        degree = 15
        translate_range = (0.1, 0.1)
        scale_range = (0.9, 1.2)
        choice = {
            # random_choice 1:
            self.augmentation_name[1]: [GaussianNoise(),
                                        RandomAffine(degrees=degree, translate=translate_range,
                                                     scale=scale_range),
                                        transforms.Compose([GaussianNoise(),
                                                            RandomAffine(degrees=degree, translate=translate_range,
                                                                         scale=scale_range)
                                                            ]
                                                           )
                                        ],
            # random_choice 2: R, S, T
            self.augmentation_name[2]: [
                RandomAffine(degrees=degree, translate=None,
                             scale=None),
                RandomAffine(degrees=0, translate=None,
                             scale=(0.8, 1.5)),
                RandomAffine(degrees=0, translate=(0.3, 0.3),
                             scale=None)
            ],
            # random_choice 3:  RT, RS, ST
            self.augmentation_name[3]: [
                RandomAffine(degrees=degree, translate=translate_range,
                             scale=None),
                RandomAffine(degrees=degree, translate=None,
                             scale=scale_range),
                RandomAffine(degrees=0, translate=translate_range,
                             scale=scale_range),
            ],
            # random_choice 4: RST
            self.augmentation_name[4]: [
                RandomAffine(degrees=degree, translate=translate_range,
                             scale=scale_range),
            ]
        }
        self.randomchoice = choice

    def set_resize(self, resize):
        self.resize = resize

    def set_transformers(self):
        self.transforms = {
            self.augmentation_name[0]: transforms.Compose([
                ToTensor()
            ]),
            self.augmentation_name[1]: transforms.Compose([
                # Choose from tranforms_list randomly
                transforms.RandomChoice(self.randomchoice['choice1']),
                ToTensor()
            ]),
            self.augmentation_name[2]: transforms.Compose([
                transforms.RandomChoice(self.randomchoice['choice2']),
                ToTensor()
            ]),
            self.augmentation_name[3]: transforms.Compose([
                transforms.RandomChoice(self.randomchoice['choice3']),
                ToTensor()
            ]),
            self.augmentation_name[4]: transforms.Compose([
                transforms.RandomChoice(self.randomchoice['choice4']),
                ToTensor()
            ])
        }

    def set_transforms_list(self):
        self.transforms_list = {
            'train':
                self.transforms,
            'val':
                self.transforms['origin']
        }

    def get_dataset(self):
        datasets = [self.dataset(
            root_dir=self.root_dir['train'],
            mode='train',
            transform=self.transforms_list['train'][r]
        )
            for r in self.augmentation_name]

        enhaced_datasets = {'train': ConcatDataset(datasets)}

        return enhaced_datasets

from torch.utils.data import ConcatDataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from preprocess import ToTensor, GaussianNoise, RandomAffine


class Stage2DataAugmentation(DataAugmentation):

    def set_transformers(self):
        self.transforms = {
            self.augmentation_name[0]: transforms.Compose([
                ToTensor()
            ]),
            self.augmentation_name[1]: transforms.Compose([
                # Choose from tranforms_list randomly
                transforms.RandomChoice(self.randomchoice['choice1']),
                ToTensor()
            ]),
            self.augmentation_name[2]: transforms.Compose([
                transforms.RandomChoice(self.randomchoice['choice2']),
                ToTensor()
            ]),
            self.augmentation_name[3]: transforms.Compose([
                transforms.RandomChoice(self.randomchoice['choice3']),
                ToTensor()
            ]),
            self.augmentation_name[4]: transforms.Compose([
                transforms.RandomChoice(self.randomchoice['choice4']),
                ToTensor()
            ])
        }

