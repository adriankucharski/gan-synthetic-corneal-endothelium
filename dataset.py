"""
Dataset

@author: Adrian Kucharski
"""
from typing import Tuple
from matplotlib import pyplot as plt

import numpy as np
from skimage import io, transform, morphology
import os
import tensorflow as tf
from hexgrid import generate_hexagons, grid_create_hexagons
from util import add_salt_and_pepper, normalization, time_measure
import json
from tensorflow.keras.models import load_model, Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_dataset(json_path: str) -> Tuple[Tuple[np.ndarray, np.ndarray]]:
    """
        Returns dataset in format Tuple[Tuple[train: np.ndarray, test: np.ndarray]]
        train | test: np.ndarray[A, B, height, width, 1]
        Where A is number of images in a fold, B is 3 - [image, gt, roi]
    """
    dataset = []
    with open(json_path, "r") as f:
        folds_json = json.load(f)
        gt_path = os.path.join(folds_json['dataset_path'], folds_json['gt'])
        images_path = os.path.join(
            folds_json['dataset_path'], folds_json['images'])
        roi_path = os.path.join(folds_json['dataset_path'], folds_json['roi'])

        def load_images(path: str) -> np.ndarray:
            image = (io.imread(os.path.join(images_path, path), as_gray=True)[
                np.newaxis, ..., np.newaxis] - 127.5) / 127.5
            gt = io.imread(os.path.join(gt_path, path), as_gray=True)[
                np.newaxis, ..., np.newaxis] / 255.0
            roi = io.imread(os.path.join(roi_path, path), as_gray=True)[
                np.newaxis, ..., np.newaxis] / 255.0
            return np.concatenate([image, gt, roi], axis=0)

        for fold in folds_json['folds']:
            dataset_fold_test = []
            dataset_fold_train = []
            for test_name in fold['test']:
                dataset_fold_test.append(load_images(test_name))
            for train_name in fold['train']:
                dataset_fold_train.append(load_images(train_name))
            dataset.append((np.array(dataset_fold_train),
                           np.array(dataset_fold_test)))
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


class HexagonDataGenerator():
    def __init__(self,
                 generator_model_path: str,
                 batch_size: int = 256,
                 hex_size: Tuple[float, float] = (17, 23),
                 neatness_range: Tuple[float, float] = (0.575, 0.7),
                 patch_size: int = 64,
                 random_shift_range: Tuple[int, int] = (1, 10),
                 sap_ratio_range: Tuple[float, float] = (0.0, 0.125),
                 salt_value_range: Tuple[float, float] = (0.5, 1.0),
                 keep_edges_tf_ratio: Tuple[float, float] = (0.5, 0.5),
                 ):
        self.model: Model = load_model(generator_model_path)
        self.batch_size = batch_size
        self.hex_size_min, self.hex_size_max = hex_size
        self.neatness_range_min, self.neatness_range_max = neatness_range
        self.patch_size = patch_size
        self.random_shift_range_min, self.random_shift_range_max = random_shift_range
        self.sap_ratio_range_min, self.sap_ratio_range_max = sap_ratio_range
        self.salt_value_range_min, self.salt_value_range_max = salt_value_range
        self.keep_edges_tf_ratio_false, self.keep_edges_tf_ratio_true = keep_edges_tf_ratio

    def _get_random_samples(self):
        hex_size = np.random.uniform(
            self.hex_size_min, self.hex_size_max, size=self.batch_size)
        neatness = np.random.uniform(
            self.neatness_range_min, self.neatness_range_max, size=self.batch_size)
        random_shift = np.random.randint(
            self.random_shift_range_min, self.random_shift_range_max, size=self.batch_size)
        sap_ratio = np.random.uniform(
            self.sap_ratio_range_min, self.sap_ratio_range_max, size=self.batch_size)
        salt_value = np.random.uniform(
            self.salt_value_range_min, self.salt_value_range_max, size=self.batch_size)
        keep_edges = np.random.choice([False, True], p=[
                                      self.keep_edges_tf_ratio_false, self.keep_edges_tf_ratio_true], size=self.batch_size)

        data = []
        for i in range(self.batch_size):
            hexagon = grid_create_hexagons(
                hex_size[i], neatness[i], self.patch_size, self.patch_size, random_shift[i])[np.newaxis, ...]
            salted_hexagon = add_salt_and_pepper(hexagon, sap_ratio[i], salt_value[i], keep_edges[i])
            z = np.random.normal(size=hexagon.shape)
            data.append(np.concatenate([salted_hexagon, z, hexagon], axis=0))
        data = np.array(data)
        generated_images = self.model.predict_on_batch([data[:, 0, ...], data[:, 1, ...]])
        return data[:, 2, ...], normalization(generated_images)

    def __iter__(self):
        self.n = 0
        self.samples_x, self.samples_y = self._get_random_samples()
        return self

    def __next__(self):
        if self.n < self.batch_size:
            self.n += 1
            return self.samples_x[self.n - 1], self.samples_y[self.n - 1]
        self.__iter__()
        return self.__next__()


if __name__ == "__main__":
    gen = HexagonDataGenerator(r'generator\models\20220325-1620\model_last.h5')
    
    # print(time_measure(lambda: [y for y in zip(range(1000), gen)]))
    for i, g in zip(range(200), gen):
        gt, image = g
        plt.imshow(np.concatenate([gt, image], axis=1), cmap="gray")
        plt.show()
