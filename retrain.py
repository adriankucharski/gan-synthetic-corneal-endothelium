"""
Retrain GAN

@author: Adrian Kucharski
"""
from auto_select import get_best_k_generators_paths
import os
from model import GAN
import datetime
from dataset import load_dataset, DataIterator
import numpy as np
from util import dumb_params
import tensorflow.keras as keras
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
    train, _ = load_dataset(r'datasets\Alizarine\folds.json')[fold]

    times = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    gen_path = None#'generator/models/20220405-2359/model_last.h5'
    disc_path = 'discriminator/models/20220405-2359/model_last.h5'

    for i in range(len(train)):
        single_image = train[i:i+1]
        validation_data = DataIterator(
            single_image, 1, patch_per_image=100, inv_values=True).get_dataset()

        gan = GAN(patch_per_image=4096,
                  g_path_save=f'generator/models/{times}/{i}',
                  d_path_save= None, #f'discriminator/models/{times}/{i}',
                  evaluate_path_save=f'data/images/{times}/{i}',
                  log_path='logs/gan/',
                  g_lr=keras.optimizers.schedules.PolynomialDecay(1e-3, decay_steps=200, power=2, end_learning_rate=1e-6),
                  d_lr=2e-5,
                  gan_lr=keras.optimizers.schedules.PolynomialDecay(1e-3,  decay_steps=100, power=2, end_learning_rate=2e-4), #1e-3
                  )
        gan.load_models(d_path=disc_path, g_path=gen_path)
        gan.train(50, single_image, evaluate_data=validation_data, save_per_epochs=1,
                  log_per_steps=4, hexagon_params=hexagon_params, dataset_rot90=False)

        path = f'data/images/{times}/{i}/**/*'
        paths, indexes, values = get_best_k_generators_paths(
            path, k=5, w=64, sigma=1, show_bar=False)
        print(i, indexes, values)
