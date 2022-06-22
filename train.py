"""
Training GAN

@author: Adrian Kucharski
"""
import os

import numpy as np

from dataset import DataIterator, load_dataset
from model import GAN
from util import dumb_params

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    hexagon_params = {
        'hexagon_size': (17, 27),
        'neatness_range': (0.55, 0.8),
        'normalize': False, 
        'inv_values': True,
        'remove_edges_ratio': 0.1,
        'rotation_range': (-60, 60),
        'random_shift': 8,
    }
    
    params = {
        'hexagon_params': hexagon_params,
        'dataset': 'datasets/Gavet/folds.json',
        'fold': 2,
        'patch_per_image': 512,
        'g_lr': 1e-5,
        'gan_lr': 2e-4,
        'd_lr':  2e-4, 
        'as_numpy': False
    }
    
    train, test = load_dataset(params['dataset'], as_numpy=params['as_numpy'])[params['fold']]
    validation_data = DataIterator(test, 1, patch_per_image=1, inv_values=True).get_dataset()


    gan = GAN(patch_per_image=params['patch_per_image'], gan_lr=params['gan_lr'])
    dumb_params(params, 'generator/hexagon_params')
    gan.train(100, train, evaluate_data=validation_data, save_per_epochs=1, hexagon_params=params['hexagon_params'])

    