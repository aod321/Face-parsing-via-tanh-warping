from model import Hybird
from dataset import Warped_HelenDataset
from preprocess import ToTensor
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from preprocess import FastTanhWarping

path = ""
model = Hybird.load_from_checkpoint(path)
root_dir = ""

transforms_list = {
    'test':
        transforms.Compose([
            ToTensor()
        ])
}
device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
# DataLoader
Dataset = Warped_HelenDataset(root_dir=root_dir,
                              mode='test',
                              transform=transforms_list['test']
                              )

dataloader = DataLoader(Dataset, batch_size=4,
                        shuffle=True, num_workers=4)

for batch in dataloader:
    image = batch['image']
    name = batch['name']
    boxes = batch['boxes']
    orig_size = batch['orig_size']
    outer, inner = model(image)
    assert outer.shape == (outer.shape[0], 1, 512, 512)
    assert inner.shape == (inner.shape[0], 4, 128, 128)

    for i in range(inner.shape[0]):
        warp_class = FastTanhWarping(boxes, orig_size[i], device=device)

        zero_label = TF.to_pil_image(np.zeros(orig_size[i]))
        for j in range(inner.shape[1]):
            inner_label = warp_class.inverse(TF.to_pil_image(inner[i][j]), output_size=orig_size[i])
            zero_label.paste(boxes)
        outer_label = warp_class.inverse(TF.to_pil_image(outer[i]), output_size=orig_size[i])
        final_label = TF.to_pil_image(np.array(outer_label) + np.array(inner_label))
        final_label.save(f"./pred_out/{name}.png", format="PNG", compress_level=0)

