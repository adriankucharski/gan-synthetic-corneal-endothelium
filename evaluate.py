import os
import sys
from glob import glob
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage.filters as filters
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import metrics
from scipy.spatial import KDTree
from skimage import morphology

from dataset import load_dataset
from predict import UnetPrediction
from util import cell_stat, neighbors_stats, postprocess_sauvola, mark_with_markers, mark_holes
from multiprocessing import Pool, Manager, Queue
import os
import itertools


def MHD(A: np.ndarray, B: np.ndarray) -> float:
    # Get all non-zero points
    setA = np.argwhere(A > 0)
    setB = np.argwhere(B > 0)
    # Build KDTree
    treeA = KDTree(setA)
    treeB = KDTree(setB)
    # Calculating the forward HD
    fhd, rhd = 0, 0
    if np.size(setA) != 0:
        fhd = np.sum(treeB.query(setA)[0]) / np.size(setA)
    # Calculating the reverse HD
    if np.size(setB) != 0:
        rhd = np.sum(treeA.query(setB)[0]) / np.size(setB)
    return np.max((fhd, rhd))


def CDA(im_pred: np.ndarray, im_true: np.ndarray, roi: np.ndarray):
    return 1
    im_pred = im_pred.reshape(im_pred.shape[:2])
    im_true = im_true.reshape(im_true.shape[:2])
    roi = roi.reshape(roi.shape[:2])
    a, _ = cell_stat(im_pred, roi)
    b, _ = cell_stat(im_true, roi)
    return (1 - (a - b) / b) * 100


def dice(A: np.ndarray, B: np.ndarray) -> float:
    A = A.flatten()
    B = B.flatten()
    return 2.0*np.sum(A * B) / (np.sum(A) + np.sum(B))


def pearsonr_image(im1: np.ndarray, im2: np.ndarray, roi: np.ndarray, markers: np.ndarray, plotpath=None) -> float:
    markers[roi == False] = 0
    markers = scipy.ndimage.label(markers)[0]
    markers = markers[..., 0] if len(markers.shape) == 3 else markers

    arr1 = mark_with_markers(im1, markers, labeled=True, roi=roi)
    arr2 = mark_with_markers(im2, markers, labeled=True, roi=roi)

    cell_size1 = []
    cell_size2 = []

    indexes = np.swapaxes(np.array(np.where(markers > 0)), 0, 1)
    for (y, x) in indexes:
        label = markers[y, x]
        cell1 = np.count_nonzero(arr1 == label)
        cell2 = np.count_nonzero(arr2 == label)
        cell_size1.append(cell1)
        cell_size2.append(cell2)

    pearsonr = scipy.stats.pearsonr(cell_size1, cell_size2)[0]
    if plotpath is not None:
        plt.clf()
        plt.scatter(cell_size1, cell_size2)
        plt.xlabel('CNN predict')
        plt.ylabel('Mask')
        plt.title('Pearsonr cells area: ' + str('%.4f' % pearsonr))
        plt.savefig(plotpath)
    return pearsonr


def cell_neighbours_stats(im1: np.ndarray, im2: np.ndarray, roi: np.ndarray, markers: np.ndarray) -> float:
    im1_n, _ = neighbors_stats(im1, markers, roi)
    im2_n, _ = neighbors_stats(im2, markers, roi)

    return metrics.accuracy_score(im1_n, im2_n)


def calculate(i: int,
              predicted: np.ndarray,
              gts: np.ndarray,
              markers: np.ndarray,
              rois: np.ndarray
              ):
    p = postprocess_sauvola(predicted[i], rois[i], size=15, pruning_op=True)

    p_dilated = morphology.dilation(
        p[..., 0], morphology.square(3))
    gt_dilated = morphology.dilation(
        gts[i][..., 0], morphology.square(3))

    mhd = MHD(gts[i], p)
    dc = dice(p_dilated, gt_dilated)
    pearsonr = pearsonr_image(gts[i], p, rois[i], markers[i])
    pearsonr_cells = cell_neighbours_stats(
        p, gts[i], rois[i], markers[i])
    cda = CDA(predicted[i], gts[i], rois[i])

    return dc, mhd, pearsonr, pearsonr_cells, cda


if __name__ == '__main__':
    datasets_names = ['Alizarine', 'Gavet', 'Hard']

    args = sys.argv[1:]
    if len(args) < 3:
        print('Provide at least four arguments')
        exit()

    dataset_name, fold, models_path = args[0:3]
    if dataset_name not in datasets_names:
        print('Dataset not found. Valid names', datasets_names)
        exit()

    stride = 16
    batch_size = 128
    _, test = load_dataset(
        f'datasets/{dataset_name}/folds.json', normalize=False, swapaxes=True)[int(fold)]
    images, gts, rois, markers = test

    for model_path in glob(os.path.join(models_path, '*')):
        unet = UnetPrediction(model_path, stride=stride, batch_size=batch_size)
        predicted = unet.predict(images)

        dcs = []
        mhds = []
        pearsonrs = []
        pearsonrs_cells = []
        cdas = []

        args = zip(range(len(predicted)),
                   itertools.repeat(predicted),
                   itertools.repeat(gts),
                   itertools.repeat(markers),
                   itertools.repeat(rois)
                   )

        with Pool(os.cpu_count() // 2) as pool:
            results = [pool.apply_async(calculate, arg) for arg in args]

            for res in results:
                dc, mhd, pearsonr, pearsonr_cells, cda = res.get()
                dcs.append(dc)
                mhds.append(mhd)
                pearsonrs.append(pearsonr)
                pearsonrs_cells.append(pearsonr_cells)
                cdas.append(cda)

        print(Path(model_path).name,
              f'{np.mean(dcs):.3f}',
              f'{np.mean(mhds):.3f}',
              f'{np.mean(pearsonrs):.3f}',
              f'{np.mean(pearsonrs_cells):.3f}',
              f'{np.mean(cdas):.3f}'
              )
