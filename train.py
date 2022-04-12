"""
Training GAN

@author: Adrian Kucharski
"""
import os
from model import GAN
from dataset import load_dataset, DataIterator
import numpy as np
from util import dumb_params
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    hexagon_params = {
        'hexagon_size': (17, 21),
        'neatness_range': (0.55, 0.7),
        'normalize': False,
        'inv_values': True,
        'remove_edges_ratio': 0.1,
        'rotation_range': (-60, 60),
        'random_shift': 8,
    }
    fold = 0
    train, test = load_dataset(r'datasets\Alizarine\folds.json')[fold]
    validation_data = DataIterator(
        test, 1, patch_per_image=1, inv_values=True).get_dataset()


    gan = GAN(patch_per_image=512)
    dumb_params(hexagon_params, 'generator/hexagon_params')
    gan.train(150, train, evaluate_data=validation_data, save_per_epochs=1, hexagon_params=hexagon_params)
