from glob import glob
from skimage import io, filters, morphology
from skimage.morphology import square
from scipy.ndimage import gaussian_filter
import os
from pathlib import Path
import numpy as np
from dataset import DataIterator, HexagonDataIterator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from util import add_salt_and_pepper, normalization
from typing import Tuple, Union
from skimage import exposure
from model import dice_loss


def generate_dataset(generator_path: str, num_of_data: int,
                     batch_size=32,
                     patch_size=64,
                     noise_size=(64, 64, 1),
                     hexagon_size=(17, 21),
                     neatness_range=(0.55, 0.7),
                     inv_values=True,
                     sap_ratio=(0.0, 0.2),
                     sap_value_range=(0.5, 1.0),
                     result_gamma_range=(0.25, 1.75),
                     noise_range=(-1e-3, 1e-3),
                     keep_edges: float = 1.0,
                     edges_thickness: int = 1,
                     remove_edges_ratio: float = 0,
                     rotate90=True,
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

    mask, image = [], []
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
        mask.extend(h)
        image.extend(p)
    mask, image = np.array(mask), np.array(image)
    image = (image + 1) / 2
    if result_gamma_range is not None:
        for i in range(len(image)):
            rgamma = np.random.uniform(*result_gamma_range)
            image[i] = exposure.adjust_gamma(image[i], rgamma)
    if noise_range is not None:
        r = np.random.uniform(*noise_range, size=image.shape)
        image = image + r
    if inv_values:
        mask = 1 - mask
    if rotate90:
        for i in range(len(image)):
            k = np.random.randint(0, 4)
            image[i], mask[i] = np.rot90(image[i], k=k), np.rot90(mask[i], k=k)
    return np.clip(image, 0, 1), mask


class UnetPrediction():
    def __init__(self, model_path: str, patch_size: int = 64, stride: int = 4, batch_size: int = 64, sigma = None):
        self.model: Model = load_model(model_path, custom_objects={'dice_loss': dice_loss})
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.sigma = sigma

    def _build_img_from_patches(self, preds: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
        patch_h, patch_w = preds.shape[1:3]

        H = (img_h-patch_h)//self.stride+1
        W = (img_w-patch_w)//self.stride+1
        prob = np.zeros((img_h, img_w, 1))
        _sum = np.zeros((img_h, img_w, 1))

        k = 0
        for h in range(H):
            for w in range(W):
                prob[h*self.stride:(h*self.stride)+patch_h, w *
                     self.stride:(w*self.stride)+patch_w, :] += preds[k]
                _sum[h*self.stride:(h*self.stride)+patch_h, w *
                     self.stride:(w*self.stride)+patch_w, :] += 1
                k += 1
        final_avg = prob/_sum
        return final_avg

    def _get_patches(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[0], img.shape[1]

        H = (h-self.patch_size)//self.stride+1
        W = (w-self.patch_size)//self.stride+1

        patches = np.empty((W*H, self.patch_size, self.patch_size, 1))
        iter_tot = 0
        for h in range(H):
            for w in range(W):
                patches[iter_tot] = (img[h*self.stride:(h*self.stride)+self.patch_size,
                                         w*self.stride:(w*self.stride)+self.patch_size, :])
                iter_tot += 1
        return patches

    def _add_outline(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        over_h = (h-self.patch_size) % self.stride
        over_w = (w-self.patch_size) % self.stride

        if (over_h != 0):
            tmp = np.zeros((h+(self.stride-over_h), w, 1))
            tmp[0:h, 0:w, :] = img
            img = tmp
        if (over_w != 0):
            tmp = np.zeros((img.shape[0], w+(self.stride - over_w), 1))
            tmp[0:img.shape[0], 0:w] = img
            img = tmp
        return img

    def _predict_from_array(self, data: Tuple[np.ndarray], verbose=0) -> Tuple[np.ndarray]:
        predicted = []
        for idx in range(len(data)):
            height, width = data[idx].shape[:2]
            img = self._add_outline(data[idx])
            if self.sigma is not None:
                img = gaussian_filter(img, self.sigma)
            new_height, new_width = img.shape[:2]
            patches = self._get_patches(img)
            predictions = self.model.predict(
                patches, batch_size=self.batch_size, verbose=verbose)
            pred_img = self._build_img_from_patches(
                predictions, new_height, new_width)

            pred_img = pred_img[:height, :width]
            predicted.append(pred_img)
        return predicted

    def _predict_from_path(self, path: Union[str, Tuple[str]], verbose=0) -> Tuple[np.ndarray]:
        images = []

        if isinstance(path, (list, tuple)):
            for im_path in path:
                images.append(io.imread(im_path, as_gray=True))
        if os.path.isfile(path):
            images.append(io.imread(path, as_gray=True))
        if os.path.isdir(path):
            for im_path in glob(os.path.join(path, '*')):
                images.append(io.imread(im_path, as_gray=True))
        for i in range(len(images)):
            if len(images[i].shape) != 3:
                images[i] = images[i][..., np.newaxis]
                if images[i].max() > 1.0:
                    images[i] = images[i] / 255.0
        return self._predict_from_array(images, verbose)

    def predict(self, data: Union[np.ndarray, Tuple[np.ndarray], str], verbose=0) -> Tuple[np.ndarray]:
        if isinstance(data, str):
            return self._predict_from_path(data, verbose)
        if isinstance(data, np.ndarray):
            if len(data.shape) == 4:
                return self._predict_from_array(data, verbose)
            elif len(data.shape) == 3:
                return self._predict_from_array([data], verbose)
        if isinstance(data, (list, tuple)) and len(data) > 0:
            if isinstance(data[0], str):
                return self._predict_from_path(data, verbose)
            if isinstance(data[0], np.ndarray):
                return self._predict_from_array(data, verbose)


if __name__ == '__main__':
    preds = []
    names = []
    image_path = 'datasets/Gavet/images/7G.png'
    for model_path in glob('segmentation/models/s/*'):
        name = Path(model_path).name
        if any([s in name for s in ['20220520-0218', '20220520-0030', '20220520-0039']]):
            model_path = os.path.join(model_path, 'model-25.hdf5')
            unet_pred = UnetPrediction(model_path,  stride=8, batch_size=128, sigma=0.75)
            print(str(Path(image_path)))
            pred = unet_pred.predict(str(Path(image_path)))[0]
            preds.append(pred)
            names.append(name)

    print(names)
    p = np.concatenate(preds, axis=1)
    plt.imshow(p, 'gray')
    plt.axis('off')
    plt.show()
