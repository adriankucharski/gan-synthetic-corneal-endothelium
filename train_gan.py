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
    
    train, _ = load_dataset(config['dataset'], as_numpy=config['as_numpy'])[config['fold']]
    validation_data = DataIterator(
        train, 1, patch_per_image=1, inv_values=True).get_dataset()

    gan = GAN(patch_per_image=config['patch_per_image'],
              gan_lr=config['gan_lr'], d_lr=config['d_lr'], g_lr=config['g_lr'])
    
    dumb_params(config, 'generator/hexagon_params') # save as logs
    
    gan.train(int(config['epochs']), train, evaluate_data=validation_data, save_per_epochs=1)
