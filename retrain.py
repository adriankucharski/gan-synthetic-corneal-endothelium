"""
Retrain GAN

@author: Adrian Kucharski
"""
import os
from model import GAN
import datetime
from dataset import load_dataset, DataIterator
import numpy as np
from util import dumb_params
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    hexagon_params = {
        'hexagon_size': (17, 24),
        'neatness_range': (0.55, 0.7),
        'normalize': False,
        'inv_values': True,
        'remove_edges_ratio': 0.1,
        'rotation_range': (-60, 60),
        'random_shift': 8,
    }
    fold = 0
    train, test = load_dataset(r'datasets\Gavet\folds.json')[fold]
    validation_data = DataIterator(
        test, 1, patch_per_image=1, inv_values=True).get_dataset()
    
    times = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    
    gen_path = ''
    disc_path = ''
    
    for i in range(len(train)):
        gan = GAN(patch_per_image=128, 
                  g_path_save=f'generator/models/{times}', 
                  d_path_save=f'discriminator/models/{times}', 
                  evaluate_path_save=f'data/images/{times}', 
                  log_path='logs/gan/'
        )
        single_image = train[i:i+1]
        gan.load_models(gen_path, disc_path)
        gan.train(20, single_image, evaluate_data=validation_data, save_per_epochs=1, hexagon_params=hexagon_params)    
    