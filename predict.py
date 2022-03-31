from skimage import io, filters, morphology
from skimage.morphology import square
import os
from pathlib import Path
import numpy as np
from dataset import load_dataset, DataIterator, HexagonDataIterator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from util import add_salt_and_pepper, normalization
from typing import Tuple


def generate_dataset(generator_path: str, num_of_data: int,
                     batch_size=32,
                     patch_size=64,
                     noise_size=(64, 64, 1),
                     hexagon_size=(17, 21),
                     neatness_range=(0.55, 0.7),
                     normalize=False,
                     inv_values=True
                     ) -> Tuple[np.ndarray, np.ndarray]:
    model_generator: Model = load_model(generator_path)
    hex_it = HexagonDataIterator(
        batch_size=batch_size,
        patch_size=patch_size,
        noise_size=noise_size,
        normalize=normalize,
        inv_values=inv_values,
        total_patches=num_of_data,
        hexagon_size=hexagon_size,
        neatness_range=neatness_range
    )

    mask, image = [], []
    for h, z in hex_it:
        # (32, 64, 64, 1)
        # (32, 64, 64, 1)
        p = model_generator.predict_on_batch([h, z])
        mask.extend(h)
        image.extend(p)
    mask, image = np.array(mask), np.array(image)
    image = (image + 1) / 2
    if inv_values:
        mask = 1 - mask
    return image, mask


if __name__ == '__main__':
    model: Model = load_model(r'generator\models\20220325-1620\model_last.h5')
    if 0:
        data = HexagonDataIterator(1, 64, 5, noise_size=(64, 64, 1))
        for x, z in data:
            # plt.imshow(x[0], cmap='gray')
            # plt.show()
            # x = morphology.dilation(x[0, ..., 0], morphology.square(2))[np.newaxis, ..., np.newaxis]
            xsap = add_salt_and_pepper(x, 0.1)
            psap = ((model.predict_on_batch([xsap, z])[0] + 1) / 2)
            p = ((model.predict_on_batch([x, z])[0] + 1) / 2)
            print(p.shape)
            x = x[0]
            xsap = xsap[0]
            xy = np.concatenate([x, p, psap, xsap], axis=1)
            plt.imshow(xy, cmap='gray')
            plt.show()
    elif 1:
        train, test = load_dataset(r'datasets\Alizarine\folds.json')[0]
        data = DataIterator(test, batch_size=1, patch_per_image=10)
        for x, y in data:
            z = np.random.normal(size=x.shape)
            xsap = add_salt_and_pepper(
                x, sap_ratio=0.05, salt_value=0.8, keep_edges=False)
            p = model.predict_on_batch([xsap, z])[0]
            x = x[0]
            y = y[0]
            xsap = xsap[0]
            p = normalization(p)
            y = normalization(y)
            plt.hist(p.flatten(), bins=100, range=(0, 1))
            plt.show()
            plt.hist(y.flatten(), bins=100, range=(0, 1))
            plt.show()
            plt.imshow(np.concatenate([xsap, x, p, y], axis=1), cmap='gray')
            plt.show()
