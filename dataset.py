import torchvision.transforms.functional as F
import numpy as np
import random
import os
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ─── Dataset Configs ──────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    'suim': {
        'classes': ('BW', 'HD', 'WR', 'RO', 'RI', 'FV'),
        'num_classes': 6,
        'palette': [
            [0, 0, 0],  # 0: BW
            [0, 0, 255],  # 1: HD
            [0, 255, 255],  # 2: WR
            [255, 0, 0],  # 3: RO
            [255, 0, 255],  # 4: RI
            [255, 255, 0],  # 5: FV
        ],
        'img_suffix': '.jpg',
        'seg_suffix': '.png',
    },
    'dut': {
        'classes': ('BW', 'SC', 'SU', 'SL', 'SF'),
        'num_classes': 5,
        'palette': [
            [0,   0,   0  ],  # 0: BW – Background/Water
            [255, 0,   0  ],  # 1: SC – Sea-floor/Coral
            [0,   255, 0  ],  # 2: SU – Sea Urchin
            [0,   0,   255],  # 3: SL – Starfish/Lobster
            [255, 255, 0  ],  # 4: SF – Sea Fan
        ],
        'img_suffix': '.jpg',
        'seg_suffix': '.png',
    },
}


def rgb_mask_to_class(mask_pil, palette):
    """
    Convert an RGB palette mask (PIL Image, mode 'RGB') into a
    single-channel class-index numpy array of shape (H, W) with dtype int64.

    Pixels that don't exactly match any palette colour are assigned
    to the closest colour by L1 distance.
    """
    mask_np = np.array(mask_pil, dtype=np.uint8)   # (H, W, 3)
    H, W, _ = mask_np.shape
    palette_np = np.array(palette, dtype=np.uint8)  # (C, 3)

    # Flatten spatial dims → (H*W, 3), compute L1 to every palette entry
    flat = mask_np.reshape(-1, 3)                   # (N, 3)
    dists = np.abs(
        flat[:, None, :].astype(np.int16) -
        palette_np[None, :, :].astype(np.int16)
    ).sum(axis=2)                                   # (N, C)

    class_map = dists.argmin(axis=1).reshape(H, W).astype(np.int64)
    return class_map


# ─── Transforms ───────────────────────────────────────────────────────────────

class ToTensor(object):
    """Convert image (PIL) → float tensor, label (np.int64 array) → long tensor."""
    def __call__(self, data):
        image, label = data['image'], data['label']
        image_tensor = F.to_tensor(image)               # (3, H, W) float
        label_tensor = torch.from_numpy(label).long()   # (H, W) long
        return {'image': image_tensor, 'label': label_tensor}


class Resize(object):
    def __init__(self, size):
        self.size = size  # (H, W) or int

    def __call__(self, data):
        image, label = data['image'], data['label']
        image = F.resize(image, self.size)
        # label is still a PIL Image at this stage (resized with NEAREST to keep class ids)
        label = F.resize(label, self.size, interpolation=InterpolationMode.NEAREST)
        return {'image': image, 'label': label}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}
        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}
        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}


class RGBMaskToClass(object):
    """
    Must be applied AFTER Resize (while label is still PIL Image in RGB mode)
    and BEFORE ToTensor.
    Converts the RGB mask PIL image to a numpy int64 class-index array.
    """
    def __init__(self, palette):
        self.palette = palette

    def __call__(self, data):
        image, label = data['image'], data['label']
        # label is a PIL Image in 'RGB' mode here
        label_np = rgb_mask_to_class(label, self.palette)
        return {'image': image, 'label': label_np}


# ─── Datasets ─────────────────────────────────────────────────────────────────

class FullDataset(Dataset):
    """
    General training/validation dataset supporting:
      - Binary segmentation (dataset_name='binary')
      - SUIM multi-class segmentation (dataset_name='suim')
      - DUT-USEG multi-class segmentation (dataset_name='dut')
    """
    def __init__(self, image_root, gt_root, size, mode, dataset_name='binary'):
        self.images = sorted([
            image_root + f for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        self.gts = sorted([
            gt_root + f for f in os.listdir(gt_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        self.dataset_name = dataset_name.lower()
        self.is_multiclass = self.dataset_name in DATASET_CONFIGS

        if self.is_multiclass:
            self.cfg = DATASET_CONFIGS[self.dataset_name]
            self.palette = self.cfg['palette']
            self.num_classes = self.cfg['num_classes']
        else:
            self.num_classes = 1  # binary

        # Build transform pipeline
        # Note: RGBMaskToClass converts PIL label → np array, so it must come
        # before ToTensor but after spatial transforms (which work on PIL).
        spatial_train = [
            Resize((size, size)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ]
        spatial_val = [Resize((size, size))]

        if self.is_multiclass:
            tensor_pipeline = [RGBMaskToClass(self.palette), ToTensor(), Normalize()]
        else:
            # Binary: keep original behaviour (label stays PIL → F.to_tensor gives float)
            tensor_pipeline = [_BinaryToTensor(), Normalize()]

        if mode == 'train':
            self.transform = _ComposeList(spatial_train + tensor_pipeline)
        else:
            self.transform = _ComposeList(spatial_val + tensor_pipeline)

    def __getitem__(self, idx):
        image = self._rgb_loader(self.images[idx])
        if self.is_multiclass:
            label = self._rgb_loader(self.gts[idx])   # load as RGB for palette matching
        else:
            label = self._binary_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def _rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class _BinaryToTensor(object):
    """Replicates original ToTensor for binary masks (PIL 'L' → float tensor)."""
    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class _ComposeList(object):
    """Like torchvision.transforms.Compose but accepts a plain list."""
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# ─── Test Dataset ─────────────────────────────────────────────────────────────

class TestDataset:
    """
    Iterable test dataset (index-based).
    Returns (image_tensor, gt_array, name) per call to load_data().
    """
    def __init__(self, image_root, gt_root, size, dataset_name='binary'):
        self.images = sorted([
            image_root + f for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        self.gts = sorted([
            gt_root + f for f in os.listdir(gt_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        self.dataset_name = dataset_name.lower()
        self.is_multiclass = self.dataset_name in DATASET_CONFIGS

        if self.is_multiclass:
            self.cfg = DATASET_CONFIGS[self.dataset_name]
            self.palette = self.cfg['palette']
            self.num_classes = self.cfg['num_classes']
        else:
            self.num_classes = 1

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self._rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        if self.is_multiclass:
            gt_pil = self._rgb_loader(self.gts[self.index])
            gt = rgb_mask_to_class(gt_pil, self.palette)   # (H, W) int64
        else:
            gt_pil = self._binary_loader(self.gts[self.index])
            gt = np.array(gt_pil)                           # (H, W) uint8

        name = self.images[self.index].split('/')[-1]
        self.index += 1
        return image, gt, name

    def _rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
