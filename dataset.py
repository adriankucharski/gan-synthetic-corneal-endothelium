"""
Dataset

@author: Adrian Kucharski
"""
import pickle
from glob import glob
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from skimage import color, io, filters
from tqdm import tqdm

from util import (I_MAX, Q_MAX)


def prepare_yiq_data(rgb_0_255_data: np.ndarray) -> np.ndarray:
    data_yiq = color.rgb2yiq(rgb_0_255_data / 255.0)
    Y = data_yiq[..., 0:1]
    I = data_yiq[..., 1:2] / I_MAX
    Q = data_yiq[..., 2:3] / Q_MAX
    return np.concatenate([Y, I, Q], axis=-1)


def prepare_yiq_dataset(paths: Tuple[str], path_save: str = None, shape=32, postprocess=None) -> Tuple[np.ndarray, np.ndarray]:
    dataset = []
    for path in tqdm(paths):
        for datapath in glob(path):
            idata = None
            try:
                idata = io.imread(datapath)[..., :3]
            except:
                idata = cv2.imread(datapath)[..., :3]
                idata = cv2.cvtColor(idata, cv2.COLOR_BGR2RGB)

            if idata.shape[-1] != 3:
                idata = color.gray2rgb(idata)
            idata = filters.gaussian(idata, 2, multichannel=True)
            idata = cv2.resize(idata, (shape, shape))
            idata = prepare_yiq_data(idata * 255)[np.newaxis, ...]
            dataset.append(idata)

    dataset = np.concatenate(np.array(dataset), axis=0)
    if postprocess is not None:
        dataset = postprocess(dataset)
    if path_save is not None:
        Path(path_save).parent.mkdir(parents=True, exist_ok=True)
        with open(path_save, 'wb') as f:
            pickle.dump((dataset[..., :1], dataset[..., 1:]), f)
    return dataset[..., :1], dataset[..., 1:]


if __name__ == '__main__':
    shape = 64
    dataset_save = Path('./datasets/dataset.h5')
    dataset_jerry_path = Path('./data/B/jerry/*.*')
    dataset_tom_path = Path('./data/B/tom/*.*')

    dataset_validation_save = Path('./datasets/dataset_validation.h5')
    dataset_validation_jerry_path = Path(
        './data/B/validation/jerry/*.*')
    dataset_validation_tom_path = Path(
        './data/B/validation/tom/*.*')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    paths_jerry = [path for path in glob(str(dataset_jerry_path))]
    paths_tom = [path for path in glob(str(dataset_tom_path))]
    paths = paths_jerry + paths_tom

    y, iq = prepare_yiq_dataset(
        paths=paths, path_save=str(dataset_save), shape=shape)
    print(y.shape, iq.shape)
    print(y.min(), y.max())
    print(iq.min(), iq.max())

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    paths_jerry = [path for path in glob(str(dataset_validation_jerry_path))]
    paths_tom = [path for path in glob(str(dataset_validation_tom_path))]
    paths = paths_jerry + paths_tom

    y, iq = prepare_yiq_dataset(
        paths=paths, path_save=str(dataset_validation_save), shape=shape)
    print(y.shape, iq.shape)
    print(y.min(), y.max())
    print(iq.min(), iq.max())
