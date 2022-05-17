"""
Other functions

@author: Adrian Kucharski
"""

import datetime
import json
import os
from datetime import timedelta
from glob import glob
from pathlib import Path
from timeit import default_timer as timer
from typing import Callable, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
import skimage.filters as filters
from matplotlib import pyplot as plt
from scipy.ndimage import measurements
from skimage import color, morphology, segmentation
from skimage.filters import threshold_sauvola
from skimage.future import graph


def dumb_params(params: dict, spath: str='segmentation/params'):
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    sargpath = os.path.join(spath, f'{time}.json')
    Path(spath).mkdir(parents=True, exist_ok=True)
    with open(sargpath, 'w') as file:
        file.write(json.dumps(params))

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


def add_jpeg_compression(im: np.ndarray, quality: int = 100) -> np.ndarray:
    if im.dtype != 'uint8':
        im = (im * 255).astype('uint8')

    _, buff = cv2.imencode(
        '.jpg', im, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    decimg: np.ndarray = cv2.imdecode(buff, cv2.IMREAD_GRAYSCALE) / 255
    return decimg.reshape(im.shape)

def pruning(im: np.ndarray, outline_val = 1.0) -> np.ndarray:
    im = np.array(im, np.int)
    elem = np.array([
        [16, 32,  64], 
        [ 8,  1, 128], 
        [ 4,  2, 256]
    ])
    val = np.array([1, 3, 5, 9, 17, 33, 65, 129, 257, 7, 13, 25, 49, 97, 193, 259, 385])
    count = True
    while count == True:
        count = False
        diff = scipy.ndimage.convolve(im, elem, mode='constant', cval=outline_val)
        diff[~np.array(im, np.bool)] = 0
        diff = np.isin(diff, val) 
        if np.any(diff>0):
            im = np.subtract(im, diff)
            count = True
    return im

def remove_small(im: np.ndarray) -> np.ndarray:
    im = im > 0
    labeled = scipy.ndimage.label(im, np.full((3,3), 1))[0]
    _, counts = np.lib.arraysetops.unique(labeled, return_counts=True)
    counts[counts == np.max(counts)] = 0
    max_idx = (np.where(counts == np.max(counts))) [0]
    im = labeled == max_idx
    return im 

def postprocess_sauvola(im: np.ndarray, roi: np.ndarray, size=5, dilation_square_size=0, pruning_op=False) -> np.ndarray:
    if len(im.shape) == 3:
        im = im[..., 0]
    if len(roi.shape) == 3:
        roi = roi[..., 0]

    im = im > threshold_sauvola(im, size, 0.2)
    roi_dil = morphology.dilation(roi, np.full((size, size), 1.0))
    im[roi_dil == False] = 0
    im = morphology.skeletonize(im)
    im = remove_small(im)
    if pruning_op:
        im = pruning(im)
    if dilation_square_size is not None and dilation_square_size > 0:
        im = morphology.dilation(im, morphology.square(dilation_square_size))
    return im.reshape((*im.shape[:2], 1))


def neighbors_stats(image: np.ndarray, markers: np.ndarray, roi: np.ndarray, show: bool=False) -> Tuple[Tuple[int], graph.RAG]:
    labels, nums = measurements.label(markers)
    
    image_labeled = np.array(image)
    for x, y in zip(*np.where(labels > 0)):
        segmentation.flood_fill(image_labeled, (x,y), labels[x,y], connectivity=1, in_place=True)
    image_labeled[roi == 0] = 0
    image_labeled = filters.median(image_labeled, morphology.square(2))
    
    g = graph.rag_mean_color(image, image_labeled)
    # Remove the first node which is connected to all nodes
    g.remove_node(0)

    if show:
        if image.shape[-1] != 3:
            image = color.gray2rgb(image)
        _, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(6, 8))
        ax[1].imshow(image_labeled)
        graph.show_rag(labels, g, image, ax=ax[0], edge_width=1.25)
        plt.tight_layout()
        plt.show()
        
    neighbors = []
    for label in range(1, nums + 1):
        n = len(list(g.neighbors(label))) if label in g.nodes else 0
        neighbors.append({label: n})

    return neighbors, g

