from predict import UnetPrediction
from glob import glob
import os
from dataset import load_dataset
import matplotlib.pyplot as plt
from skimage import filters, morphology
from scipy import spatial
import numpy as np

from util import postprocess_sauvola

def dice(A, B):
    A = A.flatten()
    B = B.flatten()
    return 2.0*np.sum(A * B) / (np.sum(A) + np.sum(B))

if __name__ == '__main__':
    fold = 0
    stride = 16
    batch_size = 128
    _, test  = load_dataset(r'datasets\Alizarine\folds.json', normalize=False, swapaxes=True)[fold]
    images, gts, rois = test

    for model_path in glob(r'segmentation\models\*'):
        model_path = os.path.join(model_path, 'model.hdf5')
        unet = UnetPrediction(model_path,  stride = stride, batch_size = batch_size)
        predicted = unet.predict(images)

        res = []
        for i in range(len(predicted)):
            gt = morphology.dilation(gts[i][..., 0], morphology.square(3))
            p = postprocess_sauvola(predicted[i], rois[i], dilation_square_size=3)
            dc = dice(p, gt)
            res.append(dc)
            
        print(model_path, np.mean(res))
