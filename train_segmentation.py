"""
Training Segmentation Unet

@author: Adrian Kucharski
"""
import datetime
import os

from skimage import transform
from model import SegmentationUnet
from dataset import load_dataset, DataIterator
from predict import generate_dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import json
from skimage import filters
import tensorflow.keras as keras
from util import dumb_params
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    generators = [r'generator\models\20220405-2359\model_144.h5', 
                  r'generator\models\20220405-2359\model_145.h5', 
                  r'generator\models\20220405-2359\model_146.h5', 
                  r'generator\models\20220405-2359\model_147.h5']
    params = {
        'generator_path': r'generator\models\20220405-2359\model_146.h5',
        'num_of_data': (256 + 64) * 20,
        'hexagon_size': (17, 21),
        'batch_size':256,
        'sap_ratio': (0.0, 0.3),
        'neatness_range': (0.6, 0.75),
        'noise_range': (-1e-2, 1e-2),
        'sap_value_range':(0.2, 0.8),
        'result_gamma_range':(0.5, 1.0), 
        'keep_edges': 0.8,
        'remove_edges_ratio': 0.05,
        'rotate90': True,
        'rotation_range': None
    }

    synthetic_image, synthetic_mask = None, None
    for generator in generators:
        params['generator_path'] = generator
        image, mask = generate_dataset(**params)
        if synthetic_mask is None and synthetic_image is None:
            synthetic_image = image
            synthetic_mask = mask
        else:
            synthetic_image = np.concatenate([synthetic_image, image], axis=0)
            synthetic_mask = np.concatenate([synthetic_mask, mask], axis=0)
    
    # synthetic_dataset = generate_dataset(**params)
    # synthetic_image, synthetic_mask = synthetic_dataset
    # # for i in range(len(synthetic_image)):
    # #     a = synthetic_image[i]
    # #     plt.imshow(a, 'gray', vmin=0, vmax=1)
    # #     plt.show()
    # # exit()
    dataset = (synthetic_image, synthetic_mask)


    fold = 0
    train, test  = load_dataset(r'datasets\Alizarine\folds.json', normalize=False)[fold]
    validation_data = DataIterator(test, 1, patch_per_image=1, inv_values=False).get_dataset()

    params['type'] = 'synthetic-dataset'
    if False:
        mask, image = DataIterator(train, 1, inv_values=False, patch_per_image=256).get_dataset()
        final_image = np.concatenate([synthetic_image, image], axis=0)
        final_mask = np.concatenate([synthetic_mask, mask], axis=0)
        dataset = (final_image, final_mask)
        params['type'] = 'half-raw-dataset'
        # dataset = (image, mask)
    if False:
        mask, image = DataIterator(train, 1, inv_values=False, patch_per_image=256 * 5).get_dataset()
        dataset = (image, mask)
        params['type'] = 'raw-dataset'
   
    unet = SegmentationUnet()
    dumb_params(params)
    unet.train(25, dataset, validation_data, validation_split=0.10)
    