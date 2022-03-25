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

# - - - - - - - - - - - - - - - - - -
#
# - - - - - - - - - - - - - - - - - -


def mapFromTo(x: np.ndarray, curr_min: float, curr_max: float, new_min: float, new_max: float) -> np.ndarray:
    y = (x-curr_min)/(curr_max-curr_min)*(new_max-new_min)+new_min
    return y


def scale(x: np.ndarray, low=-1, high=1) -> np.ndarray:
    assert low < high
    return (x - np.min(x)) * (high - low) / (np.max(x) - np.min(x)) + low

