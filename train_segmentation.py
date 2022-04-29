"""
Train Segmentation Unet

@author: Adrian Kucharski
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters

from dataset import (generate_dataset, images_preprocessing,
                     load_dataset, DataIterator)
from model import SegmentationUnet
from util import dumb_params
from auto_select import get_best_k_generators_paths
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    fold = 0
    dataset_name = 'Gavet'
    path_gens = '20220429-0021'
    
    k = 7
    _, indexes = get_best_k_generators_paths(f'data/images/{path_gens}/*', k = k, w = 64)
    print(indexes)

    indexes = [147, 148, 143, 136, 131, 102]
    
    generators = [f'generator/models/{path_gens}/model_{i}.h5' for i in indexes]
    params = {
        'num_of_data': (256) * 8,
        'hexagon_size': (17, 27),
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
            # if i in [1, 2, 3]:
            #     plt.imshow(mask[i], 'gray', vmin=0, vmax=1)
            #     plt.show()
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