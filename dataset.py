"""
Dataset

@author: Adrian Kucharski
"""
from glob import glob
from pathlib import Path
from typing import Tuple

import numpy as np
from skimage import io, morphology
import os
import tensorflow as tf
from hexgrid import generate_hexagons

def load_alizarine_dataset(path: str, mask_dilation: int = None) -> Tuple[np.ndarray]:
    'Returns Alizerine dataset with format [np.ndarray as {image, gt, roi}]'
    dataset = []
    for image_path in glob(os.path.join(path, 'images/*')):
        gt_path = image_path.replace('images', 'gt')
        roi_path = image_path.replace('images', 'roi')

        image = (io.imread(image_path, as_gray=True)[
            np.newaxis, ..., np.newaxis] - 127.5) / 127.5
        gt = io.imread(gt_path, as_gray=True) / 255.0
        roi = io.imread(roi_path, as_gray=True)[
            np.newaxis, ..., np.newaxis] / 255.0

        if mask_dilation is not None:
            gt = morphology.dilation(gt, np.ones(
                (mask_dilation, mask_dilation)))
        gt = gt[np.newaxis, ..., np.newaxis]
        dataset.append(np.concatenate([image, gt, roi], axis=0))

    return dataset

class HexagonDataIterator(tf.keras.utils.Sequence):
    def __init__(self, batch_size=32, patch_size=64, total_patches=768 * 30, noise_size=(64,)):
        """Initialization
        Dataset is (x, y, roi)"""
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.total_patches = total_patches
        self.noise_size = noise_size
        self.on_epoch_end()

    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        return len(self.h) // self.batch_size

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        'Generate one batch of data'
        # Generate indexes of the batch
        idx = np.s_[index * self.batch_size:(index+1)*self.batch_size]
        h = self.h[idx]
        z = self.z[idx]
        return h, z

    def on_epoch_end(self):
        'Generate new hexagons after one epoch'
        self.h = generate_hexagons(self.total_patches,
                                   (17, 21), 0.65, random_shift=8)
        self.z = np.random.normal(0, 1, (self.total_patches, *self.noise_size))

class DataIterator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, dataset: Tuple[np.ndarray], batch_size=32, patch_size=64, patch_per_image=768):
        """Initialization
        Dataset is (x, y, roi)"""
        self.dataset = dataset
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.patch_per_image = patch_per_image
        self.on_epoch_end()

    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        return len(self.x) // self.batch_size

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        'Generate one batch of data'
        # Generate indexes of the batch
        idx = np.s_[index * self.batch_size:(index+1)*self.batch_size]
        x = self.x[idx]
        y = self.y[idx]
        return y, x  # mask, image

    def _get_constrain_roi(self, roi: np.ndarray) -> Tuple[int, int, int, int]:
        'Get a posible patch position based on ROI'
        px, py, _ = np.where(roi != 0)
        pxy = np.dstack((px, py))[0]
        ymin, xmin = np.min(pxy, axis=0)
        ymax, xmax = np.max(pxy, axis=0)
        return [ymin, xmin, ymax, xmax]

    def on_epoch_end(self):
        'Generate new patches after one epoch'
        self.x, self.y = [], []
        mid = self.patch_size // 2
        for x, y, roi in self.dataset:
            ymin, xmin, ymax, xmax = self._get_constrain_roi(roi)
            xrand = np.random.randint(
                xmin + mid, xmax - mid, self.patch_per_image)
            yrand = np.random.randint(
                ymin + mid, ymax - mid, self.patch_per_image)

            for xpos, ypos in zip(xrand, yrand):
                self.x.append(x[ypos-mid:ypos+mid, xpos-mid:xpos+mid])
                self.y.append(y[ypos-mid:ypos+mid, xpos-mid:xpos+mid])

        self.x, self.y = np.array(self.x), np.array(self.y)
