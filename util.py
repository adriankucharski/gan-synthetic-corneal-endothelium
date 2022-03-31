"""
Other functions

@author: Adrian Kucharski
"""

from glob import glob
import os
from typing import Tuple, Union, Callable
from matplotlib import pyplot as plt

import numpy as np
from skimage import io, morphology
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import scipy.ndimage


def time_measure(routine: Callable) -> timedelta:
    start = timer()
    routine()
    end = timer()
    return timedelta(seconds=end-start)


def mapFromTo(x: np.ndarray, curr_min: float, curr_max: float, new_min: float, new_max: float) -> np.ndarray:
    y = (x-curr_min)/(curr_max-curr_min)*(new_max-new_min)+new_min
    return y


def normalization(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())


def scale(x: np.ndarray, low=-1, high=1) -> np.ndarray:
    assert low < high
    return (x - np.min(x)) * (high - low) / (np.max(x) - np.min(x)) + low


def add_salt_and_pepper(x: np.ndarray, sap_ratio: float = 0.1, salt_value: float = 0.5, keep_edges: bool = True) -> np.ndarray:
    x = np.array(x)
    sap = np.random.binomial(x.max(), sap_ratio, x.shape)
    if keep_edges:
        x[np.logical_and(sap > 0, x == 0)] = salt_value
    else:
        x[sap > 0] = salt_value
    return x


def add_marker_to_mask(mask: np.ndarray, marker_radius: int = 3, min_cell_area: int = 75) -> np.ndarray:
    nmask = np.array(mask)
    nmask[nmask > 0] = 1
    if len(nmask.shape) == 3:
        nmask = nmask[..., 0]
    dil_mask = 1 - morphology.dilation(nmask, morphology.square(3))
    labeled_array, num_features = scipy.ndimage.measurements.label(dil_mask)

    markers = np.zeros(nmask.shape)
    for i in range(1, num_features + 1):
        indexes = np.array(np.where(labeled_array == i))
        if indexes.shape[-1] >= min_cell_area:
            cx, cy = np.mean(indexes, axis=-1, dtype='int')
            markers[cx, cy] = 1
    markers = morphology.dilation(markers, morphology.disk(marker_radius))
    markers[markers > 0] = 0.5
    nmask += markers

    # nmask might has 1.5 value (nmask: 1 + markers: 0.5 =: 1.5)
    return np.clip(nmask[..., np.newaxis], 0, 1)
