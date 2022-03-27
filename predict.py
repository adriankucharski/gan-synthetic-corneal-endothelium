import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, io, morphology
from skimage.morphology import square
from tensorflow.keras.models import Model, load_model

from dataset import DataIterator, HexagonDataIterator, load_dataset
from util import add_salt_and_pepper, normalization

if __name__ == '__main__':
    model: Model = load_model(r'generator\models\20220327-2112\model_last.h5')
    if 0:
        data = HexagonDataIterator(1, 64, 5, noise_size=(64,64,1))
        for x in data:
            xsap = add_salt_and_pepper(x, 0.1)
            psap = ((model.predict_on_batch(xsap)[0] + 1) / 2)
            p = ((model.predict_on_batch(x)[0] + 1) / 2)
            print(p.shape)
            x = x[0]
            xsap = xsap[0]
            xy = np.concatenate([x,p,psap,xsap], axis=1)
            plt.imshow(xy, cmap='gray')
            plt.show()
    elif 1:
        train, test  = load_dataset(r'datasets\Alizarine\folds.json')[0]
        data = DataIterator(test, batch_size=1, patch_per_image=10)
        for x, y in data:
            xsap = add_salt_and_pepper(x, sap_ratio=0.05, salt_value=0.8, keep_edges=False)
            p = model.predict_on_batch(xsap)[0]
            x = x[0]
            y = y[0]
            xsap = xsap[0]
            p = normalization(p)
            y = normalization(y)
            plt.hist(p.flatten(), bins=100, range=(0, 1))
            plt.show()
            plt.hist(y.flatten(), bins=100, range=(0, 1))
            plt.show()
            plt.imshow(np.concatenate([xsap,x,p,y], axis=1), cmap='gray')
            plt.show()
