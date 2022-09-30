"""
Training GAN

@author: Adrian Kucharski
"""
import json
import os
from dataset import DataIterator, load_dataset
from model import GAN
from util import dumb_params

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    with open("config.json") as config_file:
        config = json.load(config_file)['gan.training']
    
    hexagon_params = config['hexagon_params']
    training_params = config['training_params']

    train, test = load_dataset(training_params['dataset'], as_numpy=training_params['as_numpy'])[training_params['fold']]
    validation_data = DataIterator(
        test, 1, patch_per_image=1, inv_values=True).get_dataset()

    gan = GAN(patch_per_image=training_params['patch_per_image'],
              gan_lr=training_params['gan_lr'], d_lr=training_params['d_lr'], g_lr=training_params['g_lr'])
    
    dumb_params(config, 'generator/hexagon_params') # save as logs
    
    gan.train(100, train, evaluate_data=validation_data,
              save_per_epochs=1, hexagon_params=hexagon_params)
