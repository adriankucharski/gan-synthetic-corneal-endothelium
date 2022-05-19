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
from util import neighbors_stats, postprocess_sauvola, mark_with_markers, mark_holes




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


def cell_stat(im: np.ndarray, mask: np.ndarray, minumum=15) -> Tuple[int, float]:
    im = mark_holes(im, mask)
    hist = np.histogram(im, np.arange(1, np.max(im) + 2),
                        (1, np.max(im) + 2))[0]
    hist = hist[hist > minumum]  # remove cells with an area less than minimum
    return (len(hist), np.mean(hist))


def dice(A: np.ndarray, B: np.ndarray) -> float:
    A = A.flatten()
    B = B.flatten()
    return 2.0*np.sum(A * B) / (np.sum(A) + np.sum(B))


def pearsonr_image(im1, im2, roi, markers, plotpath=None) -> float:
    markers[roi == False] = 0
    markers = scipy.ndimage.label(markers)[0]
    markers = markers[..., 0] if len(markers.shape) == 3 else markers

    arr1 = mark_with_markers(im1, markers, labeled=True, mask=roi)
    arr2 = mark_with_markers(im2, markers, labeled=True, mask=roi)

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


if __name__ == '__main__':
    datasets_names = ['Alizarine', 'Gavet', 'Hard']

    args = sys.argv[1:]
    if len(args) < 2:
        print('Provide at least two arguments')
        exit()

    dataset_name, action = args[0:2]
    if dataset_name not in datasets_names:
        print('Dataset not found. Valid names', datasets_names)
        exit()

    fold = 0
    stride = 16
    batch_size = 128
    _, test = load_dataset(
        f'datasets/{dataset_name}/folds.json', normalize=False, swapaxes=True)[fold]
    images, gts, rois, markers = test

    if action == 'standard':
        imgs = []
        for model_path in glob(r'segmentation\models\*'):
            name = Path(model_path).name
            if any([s in name for s in ['2330', '1133', '2236', '2205', '1900', '2018']]):
                model_path = os.path.join(model_path, 'model.hdf5')
                unet = UnetPrediction(
                    model_path,  stride=stride, batch_size=batch_size)
                predicted = unet.predict(images)

                res = []
                imgs_model = []
                for i in range(len(predicted)):
                    gt = morphology.dilation(
                        gts[i][..., 0], morphology.square(3))
                    p = postprocess_sauvola(
                        predicted[i], rois[i], dilation_square_size=3, pruning_op=True)
                    dc = dice(p, gt)
                    res.append(dc)
                    imgs_model.append(p - gt[..., np.newaxis])
                imgs.append(np.concatenate(imgs_model, axis=1))

                print(model_path, np.mean(res))

        plt.imshow(np.concatenate(imgs, axis=0), 'gray')
        plt.axis('off')
        plt.show()

    if action == 'custom':
        imgs = []
        m = 0
        for model_path in glob(r'segmentation\models\20220426-2318\model-50.hdf5'):
            m += 1
            # if m < 20:
            #     continue
            unet = UnetPrediction(
                model_path,  stride=stride, batch_size=batch_size, sigma=0.75)
            predicted = unet.predict(images)

            dcs = []
            mhds = []
            pearsonrs = []
            pearsonrs_cells = []

            for i in range(len(predicted)):
                p = postprocess_sauvola(predicted[i], rois[i], pruning_op=True)

                p_dilated = morphology.dilation(
                    p[..., 0], morphology.square(3))
                gt_dilated = morphology.dilation(
                    gts[i][..., 0], morphology.square(3))

                mhd = MHD(gts[i], p)
                dc = dice(p_dilated, gt_dilated)
                pearsonr = pearsonr_image(gts[i], p, rois[i], markers[i])
                pearsonr_cells = cell_neighbours_stats(
                    p, gts[i], rois[i], markers[i])

                dcs.append(dc)
                mhds.append(mhd)
                pearsonrs.append(pearsonr)
                pearsonrs_cells.append(pearsonr_cells)

            print('model dice mhd pearsonr cell_accuracy')
            print(Path(model_path).name, f'{np.mean(dcs):.3f}', f'{np.mean(mhds):.3f}',
                  f'{np.mean(pearsonrs):.3f}', f'{np.mean(pearsonrs_cells):.3f}')
            m += 1
