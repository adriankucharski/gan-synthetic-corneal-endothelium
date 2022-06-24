"""
Dataset

@author: Adrian Kucharski
"""
import json
import os
from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from skimage import exposure, io, morphology, transform
from tensorflow.keras.models import Model, load_model
from skimage import filters
from hexgrid import generate_hexagons, grid_create_hexagons
from util import add_salt_and_pepper, cell_stat, nonzeros_object_to_centroids, normalization
from typing import NamedTuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class GeneratorParams(NamedTuple):
    num_of_data: int
    batch_size: int = 32
    patch_size: int = 64
    noise_size: Tuple[int, int, int] = (64, 64, 1)
    hexagon_size: Tuple[int, int] = (17, 21)
    neatness_range: Tuple[float, float] = (0.55, 0.7)
    inv_values: bool = True
    sap_ratio: Tuple[float, float] = (0.0, 0.2)
    sap_value_range: Tuple[float, float] = (0.5, 1.0)
    keep_edges: float = 1.0
    edges_thickness: int = 1
    remove_edges_ratio: float = 0
    rotation_range: Tuple[float, float] = (0, 0)


def images_preprocessing(
    images: Union[Tuple[np.ndarray], np.ndarray],
    masks=None,
    gamma_range=(0.5, 1.0),
    noise_range=(-1e-2, 1e-2),
    rotate90=True,
    gaussian_sigma: float = 1.0
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    if isinstance(images, np.ndarray) and len(images.shape) == 3:
        images = [images]
    if gamma_range is not None:
        for i in range(len(images)):
            rgamma = np.random.uniform(*gamma_range)
            images[i] = exposure.adjust_gamma(images[i], rgamma)

    if noise_range is not None:
        r = np.random.uniform(*noise_range, size=images.shape)
        images = images + r

    if rotate90:
        k = np.random.randint(0, 4, size=len(images))
        for i in range(len(images)):
            if masks is not None:
                masks[i] = np.rot90(masks[i], k=k[i])
            images[i] = np.rot90(images[i], k=k[i])

    if masks is not None:
        if gaussian_sigma is not None and gaussian_sigma > 0:
            for i in range(len(masks)):
                blured = filters.gaussian(
                    masks[i], sigma=gaussian_sigma, channel_axis=-1)
                masks[i] = np.clip(blured + masks[i], 0, 1)
        return np.clip(images, 0, 1), masks

    return np.clip(images, 0, 1)

def dataset_swap_axes(dataset: Tuple):
    images, gts, rois, markers = [], [], [], []
    for i in range(len(dataset)):
        images.append(dataset[i][0])
        gts.append(dataset[i][1])
        rois.append(dataset[i][2])
        markers.append(dataset[i][3])
    return images, gts, rois, markers

def load_dataset(json_path: str, normalize=True, swapaxes=False, as_numpy = True) -> Tuple[Tuple[np.ndarray, np.ndarray]]:
    """
        json_path - path with json that describe dataset
        normalize - if true images have value [-1, 1] else [0, 1]
        swapaxes - if true [num_of_images, 4, h, w, 1] else [4, num_of_images, h, w, 1]
        Returns dataset in format Tuple[Tuple[train: np.ndarray, test: np.ndarray]]
        train | test: np.ndarray[A, B, height, width, 1]
        Where A is number of images in a fold, B is 3 - [image, gt, roi, markers]
    """
    dataset = []
    with open(json_path, "r") as f:
        folds_json = json.load(f)
        gt_path = os.path.join(folds_json['dataset_path'], folds_json['gt'])
        images_path = os.path.join(
            folds_json['dataset_path'], folds_json['images'])
        roi_path = os.path.join(folds_json['dataset_path'], folds_json['roi'])
        markers_path = os.path.join(
            folds_json['dataset_path'], folds_json['markers'])

        def load_images(path: str, w: int = None, h: int = None) -> np.ndarray:
            image = None
            if normalize:
                image = (io.imread(os.path.join(images_path, path), as_gray=True)[
                    np.newaxis, ..., np.newaxis] - 127.5) / 127.5
            else:
                image = io.imread(os.path.join(images_path, path), as_gray=True)[
                    np.newaxis, ..., np.newaxis] / 255.0
            gt = io.imread(os.path.join(gt_path, path), as_gray=True)[
                np.newaxis, ..., np.newaxis] / 255.0
            roi = io.imread(os.path.join(roi_path, path), as_gray=True)[
                np.newaxis, ..., np.newaxis] / 255.0
            markers = io.imread(os.path.join(markers_path, path), as_gray=True)[
                np.newaxis, ..., np.newaxis] / 255.0

            if w is not None and h is not None:
                w, h = int(w), int(h)
                image = transform.resize(image[0], (h, w))[np.newaxis]
                roi = transform.resize(roi[0], (h, w))[np.newaxis]
                
                gt = transform.resize(gt[0, ..., 0], (h, w), anti_aliasing=True)
                gt = morphology.skeletonize(gt > 0).astype('float')[np.newaxis, ..., np.newaxis]
                
                markers = transform.resize(markers[0, ..., 0], (h, w), anti_aliasing=True)
                markers = nonzeros_object_to_centroids(markers)[np.newaxis, ..., np.newaxis]

            return np.concatenate([image, gt, roi, markers], axis=0)

        w, h = None, None
        try:
            w, h = folds_json['width'], folds_json['height']
        except:
            pass

        for fold in folds_json['folds']:
            dataset_fold_test = []
            dataset_fold_train = []
            for test_name in fold['test']:
                dataset_fold_test.append(load_images(test_name, w, h))
            for train_name in fold['train']:
                dataset_fold_train.append(load_images(train_name, w, h))
            
            if as_numpy:
                dataset.append((np.array(dataset_fold_train), np.array(dataset_fold_test)))
            else:
                dataset.append((dataset_fold_train, dataset_fold_test))
                
    if swapaxes:
        if as_numpy:
            for i in range(len(dataset)):
                a, b = dataset[i]
                dataset[i] = (a.swapaxes(0, 1), b.swapaxes(0, 1))

    return dataset


def generate_dataset_from_generators(generators: str, params: GeneratorParams):
    synthetic_image, synthetic_mask = None, None
    for generator in generators:
        image, mask = generate_dataset(generator, **params)
        if synthetic_mask is None and synthetic_image is None:
            synthetic_image = image
            synthetic_mask = mask
        else:
            synthetic_image = np.concatenate([synthetic_image, image], axis=0)
            synthetic_mask = np.concatenate([synthetic_mask, mask], axis=0)
    return  (synthetic_image, synthetic_mask)


def generate_dataset(generator_path: str,
                     num_of_data: int,
                     batch_size=32,
                     patch_size=64,
                     noise_size=(64, 64, 1),
                     hexagon_size=(17, 21),
                     neatness_range=(0.55, 0.7),
                     inv_values=True,
                     sap_ratio=(0.0, 0.2),
                     sap_value_range=(0.5, 1.0),
                     keep_edges: float = 1.0,
                     edges_thickness: int = 1,
                     remove_edges_ratio: float = 0,
                     rotation_range=(0, 0)
                     ) -> Tuple[np.ndarray, np.ndarray]:
    model_generator: Model = load_model(generator_path)
    hex_it = HexagonDataIterator(
        batch_size=batch_size,
        patch_size=patch_size,
        noise_size=noise_size,
        inv_values=inv_values,
        total_patches=num_of_data,
        hexagon_size=hexagon_size,
        neatness_range=neatness_range,
        remove_edges_ratio=remove_edges_ratio,
        rotation_range=rotation_range
    )

    masks, images = [], []
    for h, z in hex_it:
        salty_h = h
        if sap_ratio is not None:
            salty_h = np.array(h)
            for i in range(len(salty_h)):
                ratio = np.random.uniform(*sap_ratio)
                value = np.random.uniform(*sap_value_range)
                keepe = np.random.choice(
                    [True, False], p=[keep_edges, 1 - keep_edges])
                salty_h[i] = add_salt_and_pepper(
                    salty_h[i], ratio, value, keepe)

        p = model_generator.predict_on_batch([salty_h, z])
        if edges_thickness > 1:
            for i in range(len(h)):
                func = morphology.erosion if inv_values else morphology.dilation
                h[i] = func(h[i, ..., 0], morphology.square(
                    edges_thickness))[..., np.newaxis]
        masks.extend(h)
        images.extend(p)
    masks, images = np.array(masks), np.array(images)
    images = (images + 1) / 2
    if inv_values:
        masks = 1 - masks
    return images, masks


def rescale_cell_size(
    im: np.ndarray, 
    gt: np.ndarray, 
    markers: np.ndarray, 
    roi: np.ndarray, 
    target_area: float = None, 
    precentage_scale: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert target_area is not None or precentage_scale is not None
    
    def calculate_resize_scale(area: float, target_area: float = None, precentage_scale: float = None) -> float:
        r1 = np.sqrt(area)
        if target_area is not None:
            r2 = np.sqrt(target_area)
        elif precentage_scale is not None:
            r2 = np.sqrt(area * precentage_scale)
        return r2 / r1
    
    _, area = cell_stat(gt, roi, 15)
    scale_factor = calculate_resize_scale(area, target_area, precentage_scale)
    
    im = transform.rescale(im, scale_factor)
    gt = transform.rescale(gt, scale_factor, anti_aliasing=True)
    markers = transform.rescale(markers, scale_factor, anti_aliasing=True)
    roi = transform.rescale(roi, scale_factor)
    
    gt = morphology.skeletonize(gt > 0).astype('float')
    markers = nonzeros_object_to_centroids(markers)
    
    return im, gt, markers, roi



class HexagonDataIterator(tf.keras.utils.Sequence):
    def __init__(self,
                 batch_size=32,
                 patch_size=64,
                 total_patches=32 * 24 * 30,
                 noise_size=(64, 64, 1),
                 hexagon_size=(17, 21),
                 neatness_range=(0.55, 0.70),
                 normalize=False,
                 inv_values=True,
                 remove_edges_ratio=0.1,
                 rotation_range=(0, 0),
                 random_shift=8
                 ):
        """Initialization
        Dataset is (x, y, roi)"""
        # assert total_patches % batch_size == 0
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.total_patches = total_patches
        self.noise_size = noise_size
        self.hexagon_size = hexagon_size
        self.neatness_range = neatness_range
        self.normalize = normalize
        self.inv_values = inv_values
        self.remove_edges_ratio = remove_edges_ratio
        self.rotation_range = rotation_range
        self.random_shift = random_shift
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
        neatness = np.random.uniform(*self.neatness_range)
        self.h = generate_hexagons(self.total_patches,
                                   self.hexagon_size,
                                   neatness,
                                   random_shift=self.random_shift,
                                   remove_edges_ratio=self.remove_edges_ratio,
                                   rotation_range=self.rotation_range)
        self.z = np.random.normal(0, 1, (self.total_patches, *self.noise_size))

        if self.inv_values:
            self.h = 1 - self.h

        if self.normalize:
            self.h = (self.h + 1) / 2


class DataIterator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, dataset: Tuple[np.ndarray], batch_size=32, patch_size=64, patch_per_image=768, normalize=False, inv_values=True, rot90=False):
        """
        Initialization
        Dataset is (x, y, roi)
        Inv_values - [0 - cell, 1 - edge] -> [0 - edge, 1 - cell] 
        Normalize - [0, 1] -> [-1, +1]
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.patch_per_image = patch_per_image
        self.normalize = normalize
        self.inv_values = inv_values
        self.rot90 = rot90
        self.on_epoch_end()

    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        return len(self.image) // self.batch_size

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        'Generate one batch of data and returns y, x (mask, image)'
        # Generate indexes of the batch
        idx = np.s_[index * self.batch_size:(index+1)*self.batch_size]
        x = self.image[idx]
        y = self.mask[idx]
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
        self.image, self.mask = [], []
        mid = self.patch_size // 2
        for x, y, roi, markers in self.dataset:
            ymin, xmin, ymax, xmax = self._get_constrain_roi(roi)
            xrand = np.random.randint(
                xmin + mid, xmax - mid, self.patch_per_image)
            yrand = np.random.randint(
                ymin + mid, ymax - mid, self.patch_per_image)

            for xpos, ypos in zip(xrand, yrand):
                px = x[ypos-mid:ypos+mid, xpos-mid:xpos+mid]
                py = y[ypos-mid:ypos+mid, xpos-mid:xpos+mid]
                if self.rot90:
                    k = np.random.randint(0, 3)
                    px, py = np.rot90(px, k), np.rot90(py, k)
                self.image.append(px)
                self.mask.append(py)

        self.image, self.mask = np.array(self.image), np.array(self.mask)
        if self.inv_values:
            self.mask = 1 - self.mask

        if self.normalize:
            self.image, self.mask = (self.image + 1) / 2, (self.mask + 1) / 2

    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        'self.mask, self.image'
        return self.mask, self.image


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
            salted_hexagon = add_salt_and_pepper(
                hexagon, sap_ratio[i], salt_value[i], keep_edges[i])
            z = np.random.normal(size=hexagon.shape)
            data.append(np.concatenate([salted_hexagon, z, hexagon], axis=0))
        data = np.array(data)
        generated_images = self.model.predict_on_batch(
            [data[:, 0, ...], data[:, 1, ...]])
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
    train, test = load_dataset('datasets/Rotterdam_1000/folds.json', as_numpy=False)[0]

    # print(time_measure(lambda: [y for y in zip(range(1000), gen)]))
    for imga in train:
        print(len(imga))
