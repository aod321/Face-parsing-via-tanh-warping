from torchvision import transforms
from skimage import transform as trans
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage.util import random_noise
from collections import OrderedDict


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def apply_mat_tensor(coords, matrix, device):
    matrix_tensor = torch.from_numpy(matrix.astype(np.float32)).to(device)
    if isinstance(coords, type(matrix_tensor)):
        coords = coords.type_as(matrix_tensor)
    else:
        coords = torch.from_numpy(np.array(coords, copy=False, ndmin=2))
        coords = coords.to(device)
    x, y = torch.transpose(coords, 0, 1)
    src = torch.stack([x, y, torch.ones_like(x)], dim=0)
    dst = src.T @ matrix_tensor.T
    dst[dst[:, 2] == 0, 2] = np.finfo(float).eps
    dst[:, :2] /= dst[:, 2:3]
    return dst[:, :2]


def labels2boxes(inputs):
    label_tensor = TF.to_tensor(inputs.astype(np.float32)).long()
    label_one = F.one_hot(label_tensor)
    # Shape(1, H, W, C_N=11)
    label_one = label_one.squeeze(0).permute(2, 0, 1).float()
    # Shape(11, H, W)
    leye = TF.to_pil_image(label_one[2] + label_one[4]).getbbox()
    reye = TF.to_pil_image(label_one[3] + label_one[5]).getbbox()
    nose = TF.to_pil_image(label_one[6]).getbbox()
    mouth = TF.to_pil_image(label_one[7] + label_one[8] + label_one[9]).getbbox()
    if leye is None:
        leye = [0, 0, 0, 0]
    if reye is None:
        reye = [0, 0, 0, 0]
    if nose is None:
        nose = [0, 0, 0, 0]
    if mouth is None:
        mouth = [0, 0, 0, 0]
    boxes = np.array((leye, reye, nose, mouth))
    assert boxes.shape == (4, 4)
    # Shape(4, 4)
    return boxes


class Warping(transforms.Resize):
    """
        Warping image and labels via tanh warping
    """

    def __init__(self, size, device):
        super(Warping, self).__init__(size)
        self.warp_class = None
        self.device = device

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        # labels Shape(H, W)
        boxes = labels2boxes(np.array(labels))
        # bbox format (upper_left_x, upper_left_y, down_right_x, down_right_y)
        # Shape(4, 4)
        self.warp_class = FastTanhWarping(boxes, self.size, self.device)
        # new_boxes = self._box_convert(boxes)
        warped_image = self.warp_class(image)
        warped_labels = np.array(self.warp_class(labels), dtype=np.float32, copy=False)
        warp_boxes = self._box_warp(boxes)

        sample = {'image': TF.to_tensor(warped_image),
                  'labels': TF.to_tensor(warped_labels),
                  'name': sample['name'],
                  'warp_boxes': warp_boxes,
                  'boxes': boxes,
                  'orig_size': np.array(image).shape,
                  'params': self.warp_class.tform.params
                  }
        return sample

    @staticmethod
    def _box_convert(boxes):
        # boxes Shape(4, 4)
        # (upper_x, upper_y, down_x, down_y) ---->   (cen_y, cen_x, h, w)
        # revert x, y because of the w&h revertation between PIL Image and tensor
        cen_x = (boxes[:, 0] + boxes[:, 2]) / 2.
        cen_y = (boxes[:, 1] + boxes[:, 3]) / 2.
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        new_boxes = torch.tensor((cen_y, cen_x, h, w))
        return new_boxes

    def _box_warp(self, boxes):
        new_boxes = np.tanh(self.warp_class.tform(boxes.reshape(8, 2)))
        new_boxes = torch.from_numpy(np.array(new_boxes.reshape(4, 4), dtype=np.float32, copy=False))
        assert new_boxes.shape == (4, 4)
        return new_boxes


# class SimilarityTransform_tensor(trans.SimilarityTransform):
#     def __init__(self, matrix=None, scale=None, rotation=None, translation=None, device=None):
#         super(SimilarityTransform_tensor, self).__init__(matrix, scale, rotation, translation)
#         self.device = device
#
#     # overidee _apply_mat
#     def _apply_mat(self, coords, matrix):
#         matrix_tensor = torch.from_numpy(matrix.astype(np.float32)).to(self.device)
#         if isinstance(coords, type(matrix_tensor)):
#             coords = coords.type_as(matrix_tensor)
#         else:
#             coords = torch.from_numpy(np.array(coords, copy=False, ndmin=2))
#             coords = coords.to(self.device)
#         x, y = torch.transpose(coords, 0, 1)
#         src = torch.stack([x, y, torch.ones_like(x)], dim=0)
#         dst = src.T @ matrix_tensor.T
#         dst[dst[:, 2] == 0, 2] = np.finfo(float).eps
#         dst[:, :2] /= dst[:, 2:3]
#         return dst[:, :2]


class FastTanhWarping(object):
    """
        Fast Tanh Warping implement, support CUDA.
    """

    def __init__(self, boxes, output_size, device):
        self.tform = trans.SimilarityTransform()
        self.dst = np.array([[-0.25, -0.1], [0.25, -0.1], [0.0, 0.1], [-0.15, 0.4], [0.15, 0.4]])
        self.tform2 = trans.SimilarityTransform(scale=1. / 256., rotation=0, translation=(-1, -1))
        self.landmarks = self._boxes2landmark(boxes)
        self.tform.estimate(self.landmarks, self.dst)
        self.size = output_size
        self.device = device

    def __call__(self, image):
        warped_image = self.warp(image, self.size)
        return warped_image

    def warp(self, image, output_size=None):
        if output_size is None:
            output_size = np.array(image).shape
        corrds = self._get_coords(output_size, mode='warp')
        grid = self._coords2grid(corrds, np.array(image).shape)
        warped_image = F.grid_sample(TF.to_tensor(image).unsqueeze(0).to(self.device),
                                     grid, align_corners=True)
        output = TF.to_pil_image(warped_image[0].cpu())
        return output

    def inverse(self, image, output_size=None):
        if output_size is None:
            output_size = image.shape
        corrds = self._get_coords(output_size, mode='inverse')
        grid = self._coords2grid(corrds, np.array(image).shape)
        inversed_image = F.grid_sample(TF.to_tensor(image).unsqueeze(0).to(self.device),
                                       grid, align_corners=True)
        output = TF.to_pil_image(inversed_image[0].cpu())
        return output

    def _get_coords(self, out_shape, mode='warp'):
        cols, rows = out_shape[0], out_shape[1]
        # Reshape grid coordinates into a (P, 2) array of (row, col) pairs
        tf_coords = np.indices((cols, rows), dtype=np.float32).reshape(2, -1).T
        if mode == 'warp':
            tf_coords = self._get_warped_coords(tf_coords)
        elif mode == 'inverse':
            tf_coords = self._get_inversed_coords(tf_coords)
        tf_coords = tf_coords.T.view((-1, cols, rows)).permute(0, 2, 1)
        return tf_coords

    def _get_warped_coords(self, corrds):
        matrix1 = np.linalg.inv(self.tform.params)
        matrix2 = self.tform2.params
        grid = apply_mat_tensor(atanh(apply_mat_tensor(corrds,
                                                       matrix2, self.device).clamp(-0.9999, 0.9999)
                                      ),
                                matrix1, self.device)
        return grid

    def _get_inversed_coords(self, corrds):
        # tf_inverse(artanh(tf2(c)))
        # tf2_inverse(tanh(tf(c)))
        matrix1 = self.tform.params
        matrix2 = np.linalg.inv(self.tform2.params)
        grid = apply_mat_tensor(torch.tanh(apply_mat_tensor(corrds,
                                                            matrix1, self.device)
                                           ),
                                matrix2, self.device)
        return grid

    @staticmethod
    def _boxes2landmark(boxes):
        landmarks = []
        for i in range(boxes.shape[0] - 1):
            cen_x = (boxes[i][0] + boxes[i][2]) / 2.
            cen_y = (boxes[i][1] + boxes[i][3]) / 2.
            landmarks.append((cen_x, cen_y))
        mouth1 = (boxes[3][0], boxes[3][3])
        landmarks.append(mouth1)
        mouth2 = (boxes[3][2], boxes[3][3])
        landmarks.append(mouth2)
        return np.array(landmarks)

    @staticmethod
    def _coords2grid(coords, in_image_shape):
        ih, iw = in_image_shape[0], in_image_shape[1]
        coords[0] = (2 * coords[0]) / (iw - 1) - 1
        coords[1] = (2 * coords[1]) / (ih - 1) - 1
        grid = coords.permute(1, 2, 0).unsqueeze(0)
        return grid


class RandomAffine(transforms.RandomAffine):

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        img, labels = sample['image'], sample['labels']
        warp_boxes = sample['warp_boxes']

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        img = TF.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)
        labels = TF.affine(labels, *ret, resample=self.resample, fillcolor=self.fillcolor)
        orig_box = warp_boxes * 256. + 256.

        # Affine boxes
        center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
        matrix = np.array(TF._get_inverse_affine_matrix(center, *ret)).reshape(2, 3)
        matrix = np.vstack([matrix, np.eye(3)[2]])
        assert matrix.shape == (3, 3)
        affine_trans = trans.AffineTransform(matrix=matrix)
        new_boxes = affine_trans.inverse(orig_box.reshape(-1, 2)) * (1. / 256.) - 1
        new_boxes = torch.from_numpy(new_boxes.reshape(-1, 4).astype(np.float32))

        sample.update({'image': img,
                       'labels': labels,
                       'warp_boxes': new_boxes
                       })
        return sample


class GaussianNoise(object):
    def __call__(self, sample):
        img = sample['image']
        img = np.array(img).astype(np.uint8)
        img = np.where(img != 0, random_noise(img), img)
        img = TF.to_pil_image(np.uint8(255 * img))
        sample.update({'image': img
                       })
        return sample


class ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

         Override the __call__ of transforms.ToTensor
    """

    def __call__(self, sample):
        """
                Args:
                    dict of pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

                Returns:y
                    Tensor: Converted image.
        """
        image, labels = sample['image'], sample['labels']
        image = TF.to_tensor(image)
        labels = TF.to_tensor(labels)

        sample.update({'image': image, 'labels': labels})
        return sample


class PrepareLabels(object):
    """

    """

    def __init__(self, size):
        super(PrepareLabels, self).__init__()
        self.size = size

    def __call__(self, sample):
        """
                Args:
                Returns:
        """
        labels = sample['labels']
        warp_boxes = np.array(sample['warp_boxes'] * 256. + 256.)

        np_label = np.array(labels)
        outter_labels = np.zeros(np_label.shape, dtype=np.float32)
        outter_labels[np_label == 1] = 1
        outter_labels[np_label == 10] = 10
        outter_labels = TF.to_pil_image(outter_labels)
        outter_labels = TF.resize(img=outter_labels, size=self.size, interpolation=Image.NEAREST)

        inner_labels = np.zeros(np_label.shape, dtype=np.float32)
        inner_labels[(np_label > 1) * (np_label < 10)] = np_label[(np_label > 1) * (np_label < 10)]
        inner_labels = TF.to_pil_image(inner_labels)
        inner_outs = []
        # Cropping
        for i in range(4):
            cen_x = np.floor((warp_boxes[i][0] + warp_boxes[i][2]) * 0.5)
            cen_y = np.floor((warp_boxes[i][1] + warp_boxes[i][3]) * 0.5)
            inner_outs.append(np.array(TF.crop(img=inner_labels, top=cen_x - 64,
                                               left=cen_y - 64, width=128, height=128), dtype=np.float32)
                              )
        # LEye
        inner_outs[0] = TF.to_tensor(np.where((inner_outs[0] == 2) + (inner_outs[0] == 4), inner_outs[0], 0))
        # REye
        inner_outs[1] = TF.to_tensor(np.where((inner_outs[1] == 3) + (inner_outs[1] == 5), inner_outs[1], 0))
        # Nose
        inner_outs[2] = TF.to_tensor(np.where((inner_outs[2] == 6), inner_outs[2], 0))
        # Mouth
        inner_outs[3] = TF.to_tensor(np.where((inner_outs[3] > 6) * (inner_outs[3] < 10), inner_outs[3], 0))

        new_labels = {'outer': TF.to_tensor(np.array(outter_labels, dtype=np.float32)),
                      'inner': inner_outs}

        sample.update({'parts_labels': new_labels})
        return sample
