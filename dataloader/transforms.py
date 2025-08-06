"""
Transformation and augmentations for dataset
"""
import math
import numbers
import random
import numpy as np
import torch
from PIL import Image, ImageOps

__all__ = ['Compose', 'Normalize', 'FixedResize', 'ToTensor', 'RandomSizedCrop',
           'RandomHorizontallyFlip']

class Compose():
    """Compose different transform"""
    # pylint: disable-msg=too-few-public-methods
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.is_pil_to_numpy = False

    def __call__(self, img, mask):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask = Image.fromarray(mask, mode="L")
            self.is_pil_to_numpy = True

        assert img.size == mask.size
        for aug in self.augmentations:
            img, mask = aug(img, mask)

        if self.is_pil_to_numpy:
            img, mask = np.array(img), np.array(mask, dtype=np.uint8)

        return img, mask


class Normalize():
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    # pylint: disable-msg=too-few-public-methods
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        if isinstance(img, Image.Image):
            img = np.array(img).astype(np.float32)
        if isinstance(mask, Image.Image):
            mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img, mask

class ToTensor():
    """Convert Image object in sample to Tensors."""
    # pylint: disable-msg=too-few-public-methods

    def __call__(self, img, mask):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if isinstance(img, Image.Image):
            img = np.array(img).astype(np.float32)
        if isinstance(mask, Image.Image):
            mask = np.array(mask).astype(np.float32)
        img = img.transpose((2, 0, 1))

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask


class FixedResize():
    """Resize"""
    # pylint: disable-msg=too-few-public-methods
    def __init__(self, size):
        # tuple size: (h, w)
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        img = img.resize((self.size[1], self.size[0]), Image.BILINEAR)
        mask = mask.resize((self.size[1], self.size[0]), Image.NEAREST)
        return img, mask


class RandomCrop():
    """Random crop"""
    # pylint: disable-msg=too-few-public-methods
    # pylint: disable-msg=invalid-name
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST))

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))


class CenterCrop():
    """Crop from center"""
    # pylint: disable-msg=too-few-public-methods
    # pylint: disable-msg=invalid-name
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        return (img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))


class RandomHorizontallyFlip():
    """Random Horizontal Flip"""
    # pylint: disable-msg=too-few-public-methods
    # pylint: disable-msg=invalid-name
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT))
        return img, mask


class RandomVerticallyFlip():
    """Random Vertical Flip"""
    # pylint: disable-msg=too-few-public-methods
    # pylint: disable-msg=invalid-name
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM))
        return img, mask


class RandomSizedCrop():
    """Random resize and crop"""
    # pylint: disable-msg=too-few-public-methods
    # pylint: disable-msg=invalid-name
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for _ in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize(self.size, Image.BILINEAR),
                    mask.resize(self.size, Image.NEAREST),
                )

        # Fallback
        scale = FixedResize(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))
