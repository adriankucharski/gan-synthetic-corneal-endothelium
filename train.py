"""
Training GAN

@author: Adrian Kucharski
"""
import os
from model import GAN
from dataset import load_dataset, DataIterator
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    fold = 0
    gan = GAN(patch_per_image=500)
    train, test  = load_dataset(r'datasets\Alizarine\folds.json')[fold]
    validation_data = DataIterator(test, 1, patch_per_image=2).get_dataset()
    # print(validation_data[0].shape)

    gan.train(100, train, evaluate_data=validation_data, save_per_epochs=1)