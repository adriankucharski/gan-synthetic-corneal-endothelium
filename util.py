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
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology
from skimage.segmentation import flood_fill

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
    image = image.reshape((image.shape[:2]))
    markers = markers.reshape((markers.shape[:2]))
    roi = roi.reshape((roi.shape[:2]))
    
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
        neighbors.append(n)

    return neighbors, g

def mark_with_markers(im: np.ndarray, markers: np.ndarray, labeled=False, mask: np.ndarray = None) -> np.ndarray:
    if labeled == False:
        markers = scipy.ndimage.label(markers)[0]

    im = np.array((im > 0) * (255 * 255), dtype=np.uint16)
    if mask is not None:
        im[mask == False] = (255 * 255)

    im_shape = im.shape

    im = im[..., 0] if len(im.shape) == 3 else im
    mask = mask[..., 0] if len(mask.shape) == 3 else mask
    markers = markers[..., 0] if len(markers.shape) == 3 else markers

    for (y, x) in zip(*np.where(markers > 0)):
        if im[y, x] == 255*255:
            continue
        new_value = markers[y, x]
        seed_point = (y, x)
        footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        im = flood_fill(im, seed_point, new_value, footprint=footprint)
    if mask is not None:
        im[mask == False] = 0
    im[im == 255*255] = 0
    return im[..., np.newaxis] if len(im_shape) == 3 else im


def mark_holes(im: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_e = binary_fill_holes(im > 0, np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    im = np.array((im > 0) * (255*255), np.uint16)
    x_max, y_max = im.shape
    index = np.argwhere(im > 0)

    index_full = np.array([[0, 0]])
    for x in np.arange(-1, 2):
        for y in np.arange(-1, 2):
            index_full = np.concatenate((index_full, index - [x, y]))

    T = index_full.T
    X, Y = T[0], T[1]
    # remove 0
    X, Y = X[X > 0], Y[X > 0]
    X, Y = X[Y > 0], Y[Y > 0]
    #remove < x_max
    X, Y = X[X < x_max], Y[X < x_max]
    #remove < y_max
    X, Y = X[Y < y_max], Y[Y < y_max]
    T = np.array([X, Y])
    index_full = T.T

    im[mask_e == False] = (255*255)

    i = 1
    for idx in index_full:
        x, y = idx
        if im[x, y] == 0:
            flood_fill(im, (x, y), i, selem=np.array(
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]]), inplace=True)
            i += 1
    im[mask == False] = 0
    im[mask_e == False] = 0
    im[im == (255*255)] = 0
    return im

