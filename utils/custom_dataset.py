import numpy as np
import os
from PIL import Image

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class custom_dataset:
    cmap = voc_cmap()

    def __init__(self, root, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform

        image_dir = os.path.join(self.root, 'image')
        mask_dir = os.path.join(self.root, 'mask')

        if not os.path.isdir(self.root):
            raise RuntimeError('Dataset not found or corrupted.')

        self.images = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.masks = [os.path.join(mask_dir, x) for x in os.listdir(mask_dir)]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target
    
    def __len__(self):
        return len(self.images)

    """Class for decoding mask"""
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]