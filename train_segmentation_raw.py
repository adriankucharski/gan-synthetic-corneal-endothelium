"""
Train Segmentation Unet

@author: Adrian Kucharski
"""
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters

from dataset import (generate_dataset, images_preprocessing,
                     load_dataset, DataIterator)
from model import SegmentationUnet
from util import dumb_params
from auto_select import get_best_k_generators_paths, get_best_of_the_bests
from scipy.ndimage import gaussian_filter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    params = {
        'fold': 0,
        'dataset_name': 'Rotterdam',
        'patch_per_image': 500,

        'gamma_range': (0.5, 1.0),
        'rotate90': True,
        'noise_range': None,
        'gaussian_sigma': 1.0
    }

    train, test = load_dataset(
        f'datasets/{params["dataset_name"]}/folds.json', normalize=False)[params['fold']]
    validation_data = DataIterator(
        test, 1, patch_per_image=1, inv_values=False).get_dataset()

    mask, image = DataIterator(
        train, 1, patch_per_image=params['patch_per_image'], inv_values=False).get_dataset()
    dataset = images_preprocessing(
        image, mask, 
        gamma_range=params['gamma_range'], 
        noise_range=params['noise_range'], 
        rotate90=params['rotate90'],
        gaussian_sigma=params['gaussian_sigma']
    )


    unet = SegmentationUnet(log_path_save='segmentation/logs',
                            model_path_save='segmentation/models/raw')
    dumb_params(params, 'segmentation/params/raw')
    unet.train(25, dataset, validation_data, validation_split=0.10)
    # os.system("shutdown /s /t 60")
