from skimage import io, filters, morphology
from skimage.morphology import square
import os
from pathlib import Path
import numpy as np
from model import load_alizarine_dataset, DataIterator, HexagonDataIterator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model

def add_salt_and_pepper(x: np.ndarray, sap_ratio: float = 0.1) -> np.ndarray:
    x = np.array(x)
    sap = np.random.binomial(x.max(), sap_ratio, x.shape)
    x[sap > 0] = 1.0
    return x

if __name__ == '__main__':
    model: Model = load_model(r'generator\models\20220322-2026\model_last.h5')
    model.summary()
    # datasetAlizarine = load_alizarine_dataset('datasets/Alizarine/', mask_dilation=None)
    # data = DataIterator(datasetAlizarine, batch_size=1, patch_per_image=1)

    data = HexagonDataIterator(1, 64, 5, noise_size=(64,64,1))
    
    for x, z in data:
        # plt.imshow(x[0], cmap='gray')
        # plt.show()
        # x = morphology.dilation(x[0, ..., 0], morphology.square(2))[np.newaxis, ..., np.newaxis]
        xsap = add_salt_and_pepper(x, 0.25)
        psap = ((model.predict_on_batch([xsap, z])[0] + 1) / 2.0)
        p = ((model.predict_on_batch([x, z])[0] + 1) / 2.0)
        print(p.shape)
        x = x[0]
        xsap = xsap[0]
        xy = np.concatenate([x,p,psap,xsap], axis=1)
        plt.imshow(xy, cmap='gray')
        plt.show()
        