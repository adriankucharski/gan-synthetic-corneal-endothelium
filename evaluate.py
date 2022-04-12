import os
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.ndimage.morphology import binary_fill_holes
from scipy.spatial import KDTree
from skimage import morphology
from skimage.segmentation import flood_fill

from dataset import load_dataset
from predict import UnetPrediction
from util import postprocess_sauvola


def mark_with_markers(im, markers, labeled=True, mask=None):
    if labeled == False:
        markers = scipy.ndimage.label(markers)[0]

    imb = im > 0
    im = np.array(imb * (255 * 255), dtype=np.uint16)
    if mask is not None:
        im[mask == False] = (255 * 255)
    print(markers)
    for (y, x) in zip(*np.where(markers > 0)):
        if im[y, x] == 255*255:
            continue
        im = flood_fill(im, (y, x), markers[y, x], selem=np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if mask is not None:
        im[mask == False] = 0
    #im[mask_e == False] = 0
    im[im == 255*255] = 0
    return im


def mark_holes(im, mask, size=5):
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


def MHD(A, B):
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


def cell_stat(im, mask, minumum=15):
    im = mark_holes(im, mask)
    hist = np.histogram(im, np.arange(1, np.max(im) + 2),
                        (1, np.max(im) + 2))[0]
    hist = hist[hist > minumum]  # usuń komórki mniejsze niż minumum
    return (len(hist), np.mean(hist))


def dice(A, B):
    A = A.flatten()
    B = B.flatten()
    return 2.0*np.sum(A * B) / (np.sum(A) + np.sum(B))


def pearsonr_image(im1, im2, roi, markers, plotpath=None):
    markers[roi == False] = 0
    markers = scipy.ndimage.label(markers)[0]

    im1[roi == False] = 255
    im2[roi == False] = 255
    arr1 = mark_with_markers(im1, markers)
    arr2 = mark_with_markers(im2, markers)

    cell_size1 = []
    cell_size2 = []
    for (y, x) in zip(*np.where(markers > 0)):
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

def dice(A, B):
    A = A.flatten()
    B = B.flatten()
    return 2.0*np.sum(A * B) / (np.sum(A) + np.sum(B))


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
        for model_path in glob(r'segmentation\models\20220408-1444\*'):
            unet = UnetPrediction(
                model_path,  stride=stride, batch_size=batch_size)
            predicted = unet.predict(images)

            dcs = []
            mhds = []
            pearsonrs = []
            imgs_model = []
            for i in range(len(predicted)):
                p = postprocess_sauvola(predicted[i], rois[i], pruning_op=True)
                p_dilated = morphology.dilation(p[..., 0], morphology.square(3))
                gt_dilated = morphology.dilation(
                    gts[i][..., 0], morphology.square(3))
                
                
                
                mhd = MHD(gts[i], p)
                dc = dice(p_dilated, gt_dilated)
                pearsonr = pearsonr_image(gts[i], p, rois[i], markers[i])
                
                pearsonrs.append()
                dcs.append(dc)
                mhds.append(mhd)
                pearsonrs.append(pearsonr)
                imgs_model.append(p - gt_dilated[..., np.newaxis])
            imgs.append(np.concatenate(imgs_model, axis=1))
            print(model_path, np.mean(dcs), np.mean(mhds), np.mean(pearsonrs))
