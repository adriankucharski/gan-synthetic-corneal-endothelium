from pathlib import Path
from predict import UnetPrediction
from glob import glob
import os
from dataset import load_dataset
import matplotlib.pyplot as plt
from skimage import filters, morphology
from scipy import spatial
import numpy as np
import sys

from util import postprocess_sauvola


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
    images, gts, rois = test

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
        for model_path in glob(r'segmentation\models\20220408-0025\*'):
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
