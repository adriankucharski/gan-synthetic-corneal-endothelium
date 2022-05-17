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

def replace_images_with_models(paths: Tuple[str], model_name:str = 'model_{}.h5') -> Tuple[str]:
    paths = list([p.replace('data/images', 'generator/models') for p in paths])
    for i in range(len(paths)):
        splitted = os.path.split(paths[i])
        idx = splitted[-1]
        path = os.path.join(*splitted[:-1], model_name.format(idx))
        paths[i] = path
    return paths
    
if __name__ == '__main__':
    fold = 0
    dataset_name = 'Alizarine'
    path_gens = '20220503-2026'
    
    k = 10
    paths = get_best_of_the_bests(f'data/images/{path_gens}/', k = k, w = 64, paths_only=True, skip=25)
    generators = replace_images_with_models(paths)
    
    print(generators)

    # generators = [f'generator/models/{path_gens}/model_{i}.h5' for i in indexes]
    params = {
        'num_of_data': (256) * 8,
        'hexagon_size': (17, 21),
        'batch_size':256,
        'sap_ratio': (0, 0.1),
        'neatness_range': (0.6, 0.75),
        'noise_range': (-1e-2, 1e-2),
        'sap_value_range':(0.2, 0.8),
        'result_gamma_range':(0.5, 1.0), 
        'keep_edges': 0.8,
        'remove_edges_ratio': 0.05,
        'rotate90': True,
        'rotation_range': (-60, 60),
        'inv_values': True,
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
    dataset = (synthetic_image, synthetic_mask)
    params['generators_path'] = generators
    print('synthetic_image', len(synthetic_image))
    # for i in range(len(synthetic_image)):
    #     a = synthetic_image[i]
    #     plt.imshow(a, 'gray', vmin=0, vmax=1)
    #     plt.show()
    # exit()
    
    if True:
        si, sm = dataset 
        for i in range(len(si)):
            si[i] = gaussian_filter(si[i], sigma=0.75)
        params['gaussian_filter'] = '0.75'

    train, test  = load_dataset(f'datasets/{dataset_name}/folds.json', normalize=False)[fold]
    validation_data = DataIterator(test, 1, patch_per_image=1, inv_values=False).get_dataset()

    params['type'] = 'synthetic-dataset'
    if False:
        mask, image = DataIterator(train, 1, patch_per_image=256 * 2, inv_values=False).get_dataset()
        image, mask = images_preprocessing(image, mask, gamma_range=(0.5, 1.0), noise_range=None, rotate90=True)
        dataset = (image, mask)
        params['type'] = 'raw-dataset'
        params['preprocesing'] = 'gamma_range=(0.5, 1.0), noise_range=(-1e-2, 1e-2), rotate90=True'
        final_image = np.concatenate([synthetic_image, image], axis=0)
        final_mask = np.concatenate([synthetic_mask, mask], axis=0)
        dataset = (final_image, final_mask)
        params['type'] = 'half-raw-dataset'
        print('raw-dataset', len(image))
    if False:
        mask, image = DataIterator(train, 1, patch_per_image=20480 // 20 , inv_values=False).get_dataset()
        image, mask = images_preprocessing(image, mask, gamma_range=(0.5, 1.0), noise_range=(-1e-3, 1e-3), rotate90=True)
        dataset = (image, mask)
        params['type'] = 'raw-dataset'
        params['preprocesing'] = 'gamma_range=(0.5, 1.0), noise_range=(-1e-3, 1e-3), rotate90=True'
    if True:
        image, mask = dataset
        for i in range(len(mask)):
            blured = filters.gaussian(mask[i], sigma=1, channel_axis=-1)
            mask[i] = np.clip(blured + mask[i], 0, 1)
            if i in [1, 2, 3]:
                plt.imshow(mask[i], 'gray', vmin=0, vmax=1)
                plt.show()
        dataset = (image, mask)
        params['gaussian_to_mask'] = True
    if False:
        (synthetic_image, synthetic_mask) = dataset
        for i in range(5):
            a = np.concatenate([synthetic_image[i], synthetic_mask[i]], axis=1)
            plt.imshow(a, 'gray', vmin=0, vmax=1)
            plt.show()
        exit()
    
    print('dataset', len(dataset[0]))
    params['num_of_data'] = len(dataset[0])
    unet = SegmentationUnet()
    dumb_params(params)
    unet.train(25, dataset, validation_data, validation_split=0.10)
    # os.system("shutdown /s /t 60")