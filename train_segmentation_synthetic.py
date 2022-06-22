"""
Train Segmentation Unet

@author: Adrian Kucharski
"""
import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from dataset import (generate_dataset_from_generators, images_preprocessing,
                     load_dataset, DataIterator)
from model import SegmentationUnet
from util import dumb_params

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
    # Alizerine 1
    generators = [
        r'generator\models\20220603-1720\model_114.h5',
        r'generator\models\20220603-1720\model_141.h5',
        r'generator\models\20220603-1720\model_143.h5'
    ]
    # Alizerine 2
    generators = [
        r'generator\models\20220603-1850\model_147.h5',
        r'generator\models\20220603-1850\model_145.h5',
        r'generator\models\20220603-1850\model_141.h5'
    ]
    # Rotterdam_1000 0
    generators = [
        "generator/models/20220529-2217/model_39.h5",
        "generator/models/20220529-2217/model_69.h5",
        "generator/models/20220529-2217/model_71.h5"
    ]
    # Rotterdam_1000 1
    generators = [
        r'generator\models\20220609-1841\model_15.h5',
        r'generator\models\20220609-1841\model_16.h5',
        r'generator\models\20220609-1841\model_22.h5'
    ]
    # Rotterdam_1000 2
    generators = [
        r'generator\models\20220609-2031\model_83.h5',
        r'generator\models\20220609-2031\model_34.h5',
        r'generator\models\20220609-2031\model_23.h5'
    ]
    # Gavet 0
    generators = [
        r'generator\models\20220429-0021\model_76.h5',
        r'generator\models\20220429-0021\model_81.h5',
        r'generator\models\20220429-0021\model_88.h5'
    ]
    # Gavet 1
    generators = [
        r'generator\models\20220609-2213\model_68.h5',
        r'generator\models\20220609-2213\model_97.h5',
        r'generator\models\20220609-2213\model_99.h5'
    ]
    # Gavet 2
    generators = [
        r'generator\models\20220609-2213\model_83.h5',
        r'generator\models\20220609-2213\model_94.h5',
        r'generator\models\20220609-2213\model_99.h5'
    ]

    num_of_data = 3333  # (20*256) // len(generators)
    generate_dataset_params = {
        'num_of_data': num_of_data,
        'hexagon_size': (17, 27),
        'batch_size': num_of_data // 100,
        'sap_ratio': (0.0, 0.1),
        'neatness_range': (0.6, 0.9),
        'sap_value_range': (0.2, 0.8),
        'keep_edges': 0.8,
        'remove_edges_ratio': 0.05,
        'rotation_range': (-60, 60),
        'inv_values': True,
    }
    params = {
        'fold': 2,
        'dataset_name': 'Gavet',

        'gamma_range': (0.5, 1.0),
        'rotate90': True,
        'noise_range':  (-1e-2, 1e-2),
        'gaussian_sigma': 1.0,

        'generate_dataset_params': generate_dataset_params,
        'generators': generators,
        'as_numpy': False
    }

    synthetic_image, synthetic_mask = generate_dataset_from_generators(
        params['generators'], params['generate_dataset_params'])
    dataset = images_preprocessing(synthetic_image, synthetic_mask,
                                   gamma_range=params['gamma_range'],
                                   noise_range=params['noise_range'],
                                   rotate90=params['rotate90'],
                                   gaussian_sigma=params['gaussian_sigma']
                                   )

    print(synthetic_image.shape)
    # for i in range(len(synthetic_image)):
    #     plt.imshow(np.concatenate([synthetic_image[i], synthetic_mask[i]], axis=1), 'gray')
    #     plt.show()
    # exit()

    _, test = load_dataset(
        f'datasets/{params["dataset_name"]}/folds.json', normalize=False, as_numpy=params['as_numpy'])[params['fold']]
    validation_data = DataIterator(
        test, 1, patch_per_image=1, inv_values=False).get_dataset()

    unet = SegmentationUnet(log_path_save='segmentation/logs',
                            model_path_save='segmentation/models/synthetic')
    dumb_params(params, 'segmentation/params/synthetic')
    unet.train(25, dataset, validation_data, validation_split=0.10)
    # os.system("shutdown /s /t 60")
