import os
import sys
from glob import glob
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import metrics
from scipy.spatial import KDTree
from skimage import morphology

from dataset import dataset_swap_axes, load_dataset
from predict_segmentation import UnetPrediction
from util import neighbors_stats, postprocess_sauvola, mark_with_markers
from multiprocessing import pool
import multiprocessing
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


def dice(A: np.ndarray, B: np.ndarray) -> float:
    A = A.flatten()
    B = B.flatten()
    return 2.0 * np.sum(A * B) / (np.sum(A) + np.sum(B))


def pearsonr_image(
    im1: np.ndarray,
    im2: np.ndarray,
    roi: np.ndarray,
    markers: np.ndarray,
    plotpath=None,
) -> float:
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
        plt.xlabel("CNN predict")
        plt.ylabel("Mask")
        plt.title("Pearsonr cells area: " + str("%.4f" % pearsonr))
        plt.savefig(plotpath)
    return pearsonr


def cell_neighbours_stats(
    im_true: np.ndarray, im_pred: np.ndarray, roi: np.ndarray, markers: np.ndarray
) -> Tuple[float, float]:
    neighbours_true, _ = neighbors_stats(im_true, markers, roi)
    neighbours_pred, _ = neighbors_stats(im_pred, markers, roi)

    hexagonality_true = np.sum(neighbours_true == 6) / len(neighbours_true)
    hexagonality_pred = np.sum(neighbours_pred == 6) / len(neighbours_pred)

    hexagonality = 0
    if hexagonality_true != 0:
        hexagonality = np.abs(hexagonality_true - hexagonality_pred) / hexagonality_true
    neighbours = metrics.mean_absolute_error(neighbours_true, neighbours_pred)
    return neighbours, hexagonality


def calculate(
    i: int,
    window_size: int,
    predicted: np.ndarray,
    gts: np.ndarray,
    markers: np.ndarray,
    rois: np.ndarray,
):
    p = postprocess_sauvola(predicted[i], rois[i], size=window_size, pruning_op=True)

    p_dilated = morphology.dilation(p[..., 0], morphology.square(3))
    gt_dilated = morphology.dilation(gts[i][..., 0], morphology.square(3))

    mhd = MHD(gts[i], p)
    dc = dice(p_dilated, gt_dilated)
    pearsonr = pearsonr_image(gts[i], p, rois[i], markers[i])
    neighbours, hexagonality = cell_neighbours_stats(p, gts[i], rois[i], markers[i])
    
    return dc, mhd, pearsonr, neighbours, hexagonality


if __name__ == "__main__":
    datasets_names = ["Alizarine", "Gavet", "Hard", "Rotterdam", "Rotterdam_1000"]

    args = sys.argv[1:]
    if len(args) < 4:
        print("Provide at least four arguments")
        exit()

    dataset_name, fold, models_path, window_size = args[0:4]
    if dataset_name not in datasets_names:
        print("Dataset not found. Valid names", datasets_names)
        exit()

    stride = 8
    batch_size = 256
    standardization = False
    _, test = load_dataset(
        f"datasets/{dataset_name}/folds.json", normalize=False, as_numpy=False
    )[int(fold)]

    images, gts, rois, markers = dataset_swap_axes(test)
    print('Number of images:', len(images))
    
    suffix = ""
    if "raw" in models_path:
        suffix = "raw"
    elif "synthetic" in models_path:
        suffix = "synthetic"
    elif "mix" in models_path:
        suffix = "mix"

    filename = f"{dataset_name}_{fold}_{window_size}_{suffix}.txt"
    [Path(path).mkdir(exist_ok=True) for path in ["result_best", "result"]]
    with open(os.path.join("result", filename), "w") as rf, open(
        os.path.join("result_best", filename), "w"
    ) as rb:
        p = (
            models_path
            if os.path.isfile(models_path)
            else os.path.join(models_path, "*")
        )

        cpus = multiprocessing.cpu_count()
        print(p, os.path.isfile(models_path), f"| CPU: {cpus}")
        thread_pool = pool.ThreadPool(cpus)
        for model_path in glob(p)[10:]:

            unet = UnetPrediction(
                model_path,
                stride=stride,
                batch_size=batch_size,
                standardization=standardization,
            )
            predicted = unet.predict(images)

            dcs = []
            mhds = []
            pearsonrs = []
            neighbours = []
            hexagonality = []

            args = zip(
                range(len(predicted)),
                itertools.repeat(int(window_size)),
                itertools.repeat(predicted),
                itertools.repeat(gts),
                itertools.repeat(markers),
                itertools.repeat(rois),
            )

            results = thread_pool.starmap(calculate, args)
            for res in results:
                dc, mhd, pearsonr, neighbours_mse, hexagonality_rate = res
                dcs.append(dc)
                mhds.append(mhd)
                pearsonrs.append(pearsonr)
                neighbours.append(neighbours_mse)
                hexagonality.append(hexagonality_rate)

            print(
                Path(model_path).name,
                f"{np.mean(dcs):.3f}",
                f"{np.mean(mhds):.3f}",
                f"{np.mean(pearsonrs):.3f}",
                f"{np.mean(neighbours):.3f}",
                f"{np.mean(hexagonality):.3f}",
            )

            line = "\n".join(
                [
                    f"{np.mean(dcs):.3f}",
                    f"{np.mean(mhds):.3f}",
                    f"{np.mean(pearsonrs):.3f}",
                    f"{np.mean(neighbours):.3f}",
                    f"{np.mean(hexagonality):.3f}",
                ]
            )
            rf.write(line + "\n")
            line = (
                "\n".join(
                    [
                        f"dice, {dcs}",
                        f"mhd, {mhds}",
                        f"pearsonrs, {pearsonrs}",
                        f"neighbours, {neighbours}",
                        f"hexagonality, {hexagonality}",
                    ]
                )
                .replace("[", "")
                .replace("]", "")
            )
            rb.write(line + "\n")
