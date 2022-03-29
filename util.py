"""
Other functions

@author: Adrian Kucharski
"""

from glob import glob
import os
from typing import Tuple, Union, Callable

import numpy as np
from skimage import io
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np

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

def add_salt_and_pepper(x: np.ndarray, sap_ratio: float = 0.1, salt_value: float = 0.5, keep_edges: bool=True) -> np.ndarray:
    x = np.array(x)
    sap = np.random.binomial(x.max(), sap_ratio, x.shape)
    if keep_edges:
        x[np.logical_and(sap > 0, x == 0)] = salt_value
    else:
        x[sap > 0] = salt_value
    return x
