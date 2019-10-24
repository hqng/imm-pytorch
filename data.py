"""
Load CelebA dataset.
Perform proc_img_pair (crop,resize) and tps_warping
"""
import os
from os import path
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F
from torch.utils import data

from torchvision import transforms as T
from tps_sampler import TPSRandomSampler


#------------------------------------------------------------------------------
#Initial Dataset
#------------------------------------------------------------------------------

def load_dataset(data_root, dataset, subset):
    image_dir = os.path.join(data_root, 'celeba', 'img_align_celeba')

    with open(os.path.join(data_root, 'celeba', 'list_landmarks_align_celeba.txt'), 'r') as f:
        lines = f.read().splitlines()
    # skip header
    lines = lines[2:]
    image_files = []
    keypoints = []
    for line in lines:
        image_files.append(line.split()[0])
        keypoints.append([int(x) for x in line.split()[1:]])
    keypoints = np.array(keypoints, dtype=np.float32)
    assert image_files[0] == '000001.jpg'

    images_set = np.zeros(len(image_files), dtype=np.int32)

    if dataset == 'celeba':
        with open(os.path.join(data_root, 'celeba', 'list_eval_partition.txt'), 'r') as f:
            celeba_set = [int(line.split()[1]) for line in f.readlines()]
        images_set[:] = celeba_set
        images_set += 1

    if dataset == 'celeba':
        if subset == 'train':
            label = 1
        elif subset == 'val':
            label = 2
        else:
            raise ValueError(
                'subset = %s for celeba dataset not recognized.' % subset)

    image_files = np.array(image_files)
    images = image_files[images_set == label]
    keypoints = keypoints[images_set == label]

    # convert keypoints to
    # [[lefteye_x, lefteye_y], [righteye_x, righteye_y], [nose_x, nose_y],
    #  [leftmouth_x, leftmouth_y], [rightmouth_x, rightmouth_y]]
    keypoints = np.reshape(keypoints, [-1, 5, 2])

    return image_dir, images, keypoints


class DatasetFromFolder(data.Dataset):
    """Manipulate data from folder
    """
    def __init__(self, data_root, dataset, subset, transform):
        super(DatasetFromFolder, self).__init__()
        self.transform = transform
        self.image_dir, self.image_name, self.keypoints = load_dataset(data_root, dataset, subset)
        # len = image_name.shape[0] // 5 #use 20% of data
        # self.image_name = image_name[:len]
        # self.keypoints = keypoints[:len]

    def __getitem__(self, idx):
        img = Image.open(path.join(self.image_dir, self.image_name[idx]))
        img = self.transform(img)
        keypts = torch.from_numpy(self.keypoints[idx])
        return img, keypts

    def __len__(self):
        return self.image_name.shape[0]


def transforms(size=[128, 128]):
    return T.Compose([
        # T.Resize(size),
        T.ToTensor(),
    ])


def get_dataset(data_root, dataset, subset):
    return DatasetFromFolder(data_root, dataset, subset, transform=transforms())

#------------------------------------------------------------------------------
#Get method (used for DataLoader)
#------------------------------------------------------------------------------

class BatchTransform(object):
    """ Preprocessing batch of pytorch tensors
    """
    def __init__(self, image_size=[128, 128], \
            rotsd=[0.0, 5.0], scalesd=[0.0, 0.1], \
            transsd=[0.1, 0.1], warpsd=[0.001, 0.005, 0.001, 0.01]):
        self.image_size = image_size
        self.target_sampler, self.source_sampler = \
            self._create_tps(image_size, rotsd, scalesd, transsd, warpsd)

    def exe(self, image, landmarks=None):
        #call _proc_im_pair
        batch = self._proc_im_pair(image, landmarks=landmarks)

        #call _apply_tps
        image, future_image, future_mask = self._apply_tps(batch['image'], batch['mask'])

        batch.update({'image': image, 'future_image': future_image, 'mask': future_mask})

        return batch

    #TPS
    def _create_tps(self, image_size, rotsd, scalesd, transsd, warpsd):
        """create tps sampler for target and source images"""
        target_sampler = TPSRandomSampler(
            image_size[1], image_size[0], rotsd=rotsd[0],
            scalesd=scalesd[0], transsd=transsd[0], warpsd=warpsd[:2], pad=False)
        source_sampler = TPSRandomSampler(
            image_size[1], image_size[0], rotsd=rotsd[1],
            scalesd=scalesd[1], transsd=transsd[1], warpsd=warpsd[2:], pad=False)
        return target_sampler, source_sampler

    def _apply_tps(self, image, mask):
        #expand mask to match batch size and n_dim
        mask = mask[None, None].expand(image.shape[0], -1, -1, -1)
        image = torch.cat([mask, image], dim=1)
        # shape = image.shape

        future_image = self.target_sampler.forward(image)
        image = self.source_sampler.forward(future_image)

        #reshape -- no need
        # image = image.reshape(shape)
        # future_image = future_image.reshape(shape)

        future_mask = future_image[:, 0:1, ...]
        future_image = future_image[:, 1:, ...]

        mask = image[:, 0:1, ...]
        image = image[:, 1:, ...]

        return image, future_image, future_mask

    #Process image pair
    def _proc_im_pair(self, image, landmarks=None):
        m, M = image.min(), image.max()

        height, width = self.image_size[:2]

        #crop image
        crop_percent = 0.8
        final_sz = self.image_size[0]
        resize_sz = np.round(final_sz / crop_percent).astype(np.int32)
        margin = np.round((resize_sz - final_sz) / 2.0).astype(np.int32)

        if landmarks is not None:
            original_sz = image.shape[-2:]
            landmarks = self._resize_points(
                landmarks, original_sz, [resize_sz, resize_sz])
            landmarks -= margin

        image = F.interpolate(image, \
            size=[resize_sz, resize_sz], mode='bilinear', align_corners=True)

        #take center crop
        image = image[..., margin:margin + final_sz, margin:margin + final_sz]
        image = torch.clamp(image, m, M)

        mask = self._get_smooth_mask(height, width, 10, 20) #shape HxW
        mask = mask.to(image.device)

        future_landmarks = landmarks
        # future_image = image.clone()

        batch = {}
        batch.update({'image': image, 'mask': mask, \
            'landmarks': landmarks, 'future_landmarks': future_landmarks})

        return batch

    def _resize_points(self, points, size, new_size):
        dtype = points.dtype
        device = points.device

        size = torch.tensor(size).to(device).float()
        new_size = torch.tensor(new_size).to(device).float()

        ratio = new_size / size
        points = (points.float() * ratio[None]).type(dtype)
        return points

    def _get_smooth_step(self, n, b):
        x = torch.linspace(-1, 1, n)
        y = 0.5 + 0.5 * torch.tanh(x / b)
        return y

    def _get_smooth_mask(self, h, w, margin, step):
        b = 0.4
        step_up = self._get_smooth_step(step, b)
        step_down = self._get_smooth_step(step, -b)

        def _create_strip(size):
            return torch.cat(
                [torch.zeros(margin),
                step_up,
                torch.ones(size - 2 * margin - 2 * step),
                step_down,
                torch.zeros(margin)], dim=0)

        mask_x = _create_strip(w)
        mask_y = _create_strip(h)
        mask2d = mask_y[:, None] * mask_x[None]
        return mask2d
