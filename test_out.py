from model import Hybird
from dataset import Warped_HelenDataset
from preprocess import ToTensor
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from preprocess import FastTanhWarping
import argparse
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=9, type=int, help="Which GPU to train.")
parser.add_argument('--pretrain_path', metavar='DIR', default="exp_stage1/checkpoints8625eff6/best.pth.tar", type=str,
                    help='path to save output')

args = parser.parse_args()
print(args)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = Hybird(args).to(device)

path = "/home/yinzi/data4/fcn_res/exp_stage2/checkpoints7fc8d2bc/best.pth.tar"
state = torch.load(path, map_location=device)
model.load_state_dict(state['model'])
print(path + " loaded")

root_dir = {
    'test': "/home/yinzi/data4/warped/test"
}

transforms_list = {
    'test':
        transforms.Compose([
            ToTensor()
        ])
}
# DataLoader
Dataset = Warped_HelenDataset(root_dir=root_dir['test'],
                              mode='test',
                              transform=transforms_list['test']
                              )

dataloader = DataLoader(Dataset, batch_size=16,
                        shuffle=True, num_workers=4)

for batch in dataloader:
    image = batch['image'].to(device)
    name = batch['name']
    boxes = batch['boxes']
    warp_box = batch['warp_boxes']
    N = image.shape[0]
    inverse_box = torch.round(warp_box * 256. + 256.).float().numpy()
    assert boxes.shape == (N, 4, 4)
    orig_size = batch['orig_size']
    inner, outer = model(image)

    name_list = ['leye', 'reye', 'nose', 'mouth']
    for x in name_list:
        inner[x] = F.softmax(inner[x], dim=1).argmax(dim=1, keepdim=False
                                                     ).cpu().numpy().astype(np.uint8)

    outer_label = F.interpolate(outer, size=(512, 512), mode='bilinear', align_corners=True)

    outer_label = F.softmax(outer_label, dim=1).argmax(dim=1, keepdim=False
                                                       ).cpu().numpy().astype(np.uint8)
    for i in range(N):
        warp_class = FastTanhWarping(boxes[i], orig_size[i], device=device)
        zero_label = TF.to_pil_image(np.zeros((512, 512)).astype(np.uint8))
        # for j in range(4):
        #     inner_label = TF.to_pil_image(inner[name_list[j]][i])
        #     zero_label.paste(inner_label, inverse_box[i])
        #
        # final_label = (TF.to_tensor(zero_label) + outer_label).numpy()
        # final_label = warp_class.inverse(final_label, output_size=orig_size[i])
        # os.makedirs("./pred_out", exist_ok=True)
        # final_label.save(f"./pred_out/{name[i]}.png", format="PNG", compress_level=0)
