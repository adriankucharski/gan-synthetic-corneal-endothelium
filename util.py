"""
Other functions

@author: Adrian Kucharski
"""

from glob import glob
import os
from typing import Tuple, Union

import numpy as np
from skimage import io
I_MAX = 0.5957
Q_MAX = 0.5226


def imsread(paths: Union[str, Tuple[str]], as_gray: bool = False, as_float: bool = False) -> Tuple[np.ndarray]:
    if isinstance(paths, str) and (os.path.isdir(paths) or paths.endswith('*')):
        if paths.endswith('*'):
            return imsread(glob(paths), as_gray)
        return imsread(tuple(glob(os.path.join(paths, '*'))), as_gray, as_float)
    if isinstance(paths, str) and os.path.isfile(paths):
        im = io.imread(paths, as_gray)
        if as_float:
            im = im / 255.0
        if im.shape[-1] > 4:
            return [im]
        return [im[..., :3]]
    if isinstance(paths, tuple) or isinstance(paths, list):
        return [imsread(path, as_gray, as_float)[0] for path in paths]


def crop_image(im: np.ndarray, labels: list):
    H, W, _ = im.shape
    _, x, y, w, h = labels
    tindices = [int((x - w / 2) * W),
                int((y - h / 2) * H), int(w*W), int(h*H)]
    tx, ty, tw, th = tindices
    part = im[ty:ty+th, tx:tx+tw]
    return part, tindices


def rgb2luminosity(rgb: np.ndarray, as_uint8=False) -> np.ndarray:
    coeffs = np.array([0.299, 0.587, 0.114], dtype=rgb.dtype)
    luminosity = rgb @ coeffs
    return (luminosity * 255).astype('uint8') if as_uint8 else luminosity

# - - - - - - - - - - - - - - - - - -
#
# - - - - - - - - - - - - - - - - - -


def mapFromTo(x: np.ndarray, curr_min: float, curr_max: float, new_min: float, new_max: float) -> np.ndarray:
    y = (x-curr_min)/(curr_max-curr_min)*(new_max-new_min)+new_min
    return y


def scale(x: np.ndarray, low=-1, high=1) -> np.ndarray:
    assert low < high
    return (x - np.min(x)) * (high - low) / (np.max(x) - np.min(x)) + low


# - - - - - - - - - - - - - - - - - -
# YIQ Operation
# - - - - - - - - - - - - - - - - - -

def iq_from_tanh(iq: np.ndarray) -> np.ndarray:
    i, q = iq[..., :1] * I_MAX, iq[..., 1:] * Q_MAX
    iq = np.concatenate([i, q], axis=-1)
    return iq


def tanh_from_iq(iq: np.ndarray) -> np.ndarray:
    i, q = iq[..., :1] / I_MAX, iq[..., 1:] / Q_MAX
    iq = np.concatenate([i, q], axis=-1)
    return iq


def tanh_from_yiq(yiq: np.ndarray) -> np.ndarray:
    y, iq = yiq[..., :1], yiq[..., 1:]
    i, q = iq[..., :1] / I_MAX, iq[..., 1:] / Q_MAX
    yiq = np.concatenate([y, i, q], axis=-1)
    return yiq


def yiq_from_tanh(yiq: np.ndarray) -> np.ndarray:
    y, iq = yiq[..., :1], yiq[..., 1:]
    iq = iq_from_tanh(iq)
    return np.concatenate([y, iq], axis=-1)


def concat_yiq(lumination: np.ndarray, iq: np.ndarray, from_tanh=True) -> np.ndarray:
    if from_tanh:
        iq = iq_from_tanh(iq)
    return np.concatenate([lumination, iq], axis=-1)
