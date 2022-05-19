"""
Train Segmentation Unet

@author: Adrian Kucharski
"""
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
from sklearn import datasets

from dataset import (generate_dataset_from_generators, images_preprocessing,
                     load_dataset, DataIterator)
from model import SegmentationUnet
from util import dumb_params
from auto_select import get_best_k_generators_paths, get_best_of_the_bests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def replace_images_with_models(paths: Tuple[str], model_name: str = 'model_{}.h5') -> Tuple[str]:
    paths = list([p.replace('data/images', 'generator/models') for p in paths])
    for i in range(len(paths)):
        splitted = os.path.split(paths[i])
        idx = splitted[-1]
        path = os.path.join(*splitted[:-1], model_name.format(idx))
        paths[i] = path
    return paths


if __name__ == '__main__':
    generate_dataset_params = {
        'num_of_data': (256) * 8,
        'hexagon_size': (17, 21),
        'batch_size': 256,
        'sap_ratio': (0, 0.1),
        'neatness_range': (0.6, 0.8),
        'sap_value_range': (0.2, 0.8),
        'keep_edges': 0.8,
        'remove_edges_ratio': 0.05,
        'rotation_range': (-60, 60),
        'inv_values': True,
    }
    params = {
        'fold': 0,
        'dataset_name': 'Alizarine',
        'patch_per_image': 500,

        'gamma_range': (0.5, 1.0),
        'rotate90': True,
        'noise_range': None,
        'gaussian_sigma': 1.0,

        'path_gens': '20220503-2026',
        'best_k': 1,

        'generate_dataset_params': generate_dataset_params,
    }

    # paths = get_best_of_the_bests(
    #     f'data/images/{params["path_gens"]}/', k=params['best_k'], w=64, paths_only=True, skip=25)
    # generators = replace_images_with_models(paths)
    generators = ['generator/models/20220503-2026\\4\\20220503-2122\\model_29.h5']

    synthetic_image, synthetic_mask = generate_dataset_from_generators(
        generators, params['generate_dataset_params'])
    dataset = images_preprocessing(synthetic_image, synthetic_mask,
                                   gamma_range=params['gamma_range'],
                                   noise_range=params['noise_range'],
                                   rotate90=params['rotate90'],
                                   gaussian_sigma=params['gaussian_sigma']
                                   )


    for i in range(len(synthetic_image)):
        a = synthetic_image[i]
        plt.imshow(a, 'gray', vmin=0, vmax=1)
        plt.show()
    exit()

    _, test = load_dataset(
        f'datasets/{dataset_name}/folds.json', normalize=False)[fold]
    validation_data = DataIterator(
        test, 1, patch_per_image=1, inv_values=False).get_dataset()

    unet = SegmentationUnet()
    dumb_params(params)
    unet.train(25, dataset, validation_data, validation_split=0.10)
    # os.system("shutdown /s /t 60")
