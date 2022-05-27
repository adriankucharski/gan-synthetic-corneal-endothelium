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
    # Rotterdam
    generators = [
        # "generator/models/20220522-0151/model_16.h5",
        "generator/models/20220522-0151/model_23.h5",
        "generator/models/20220522-0151/model_113.h5",
        # "generator/models/20220522-0151/model_116.h5"
    ]
    
    # Alizarine    
    # generators = [
    #     'generator/models/20220405-2359/model_149.h5',
    #     'generator/models/20220405-2359/model_145.h5',
    #     'generator/models/20220405-2359/model_144.h5',
    #     'generator/models/20220405-2359/model_137.h5',
    #     'generator/models/20220405-2359/model_136.h5',
    #     'generator/models/20220405-2359/model_97.h5',
    #     'generator/models/20220405-2359/model_91.h5',
    #     'generator/models/20220405-2359/model_85.h5',
    #     'generator/models/20220405-2359/model_81.h5',
    #     'generator/models/20220405-2359/model_77.h5',
    # ]
    
    # Gavet
    # generators = [
    #     'generator/models/20220429-0021/model_148.h5',
    #     'generator/models/20220429-0021/model_147.h5',
    #     'generator/models/20220429-0021/model_145.h5',
    #     'generator/models/20220429-0021/model_143.h5',
    #     'generator/models/20220429-0021/model_120.h5',
    #     'generator/models/20220429-0021/model_109.h5',
    #     'generator/models/20220429-0021/model_102.h5',
    #     'generator/models/20220429-0021/model_118.h5',
    # ]

    generate_dataset_params = {
        'num_of_data': 1024 * 20 // len(generators),
        'hexagon_size': (25, 32),
        'batch_size': 256,
        'sap_ratio': (0, 0.1),
        'neatness_range': (0.7, 0.8),
        'sap_value_range': (0.2, 0.4),
        'keep_edges': 0.8,
        'remove_edges_ratio': 0.05,
        'rotation_range': (-60, 60),
        'inv_values': True,
    }
    params = {
        'fold': 0,
        'dataset_name': 'Rotterdam',

        'gamma_range': (0.5, 1.0),
        'rotate90': True,
        'noise_range': (-1e-2, 1e-2),
        'gaussian_sigma': 1.0,

        'generate_dataset_params': generate_dataset_params,
        'generators': generators,
    }

    synthetic_image, synthetic_mask = generate_dataset_from_generators(
        params['generators'], params['generate_dataset_params'])
    dataset = images_preprocessing(synthetic_image, synthetic_mask,
                                   gamma_range=params['gamma_range'],
                                   noise_range=params['noise_range'],
                                   rotate90=params['rotate90'],
                                   gaussian_sigma=params['gaussian_sigma']
                                   )

    # for i in range(len(synthetic_image)):
    #     plt.imshow(np.concatenate([synthetic_image[i], synthetic_mask[i]], axis=1), 'gray')
    #     plt.show()
    # exit()

    _, test = load_dataset(
        f'datasets/{params["dataset_name"]}/folds.json', normalize=False)[params['fold']]
    validation_data = DataIterator(
        test, 1, patch_per_image=1, inv_values=False).get_dataset()

    unet = SegmentationUnet(log_path_save='segmentation/logs',
                            model_path_save='segmentation/models/synthetic')
    dumb_params(params, 'segmentation/params/synthetic')
    unet.train(25, dataset, validation_data, validation_split=0.10)
    # os.system("shutdown /s /t 60")
