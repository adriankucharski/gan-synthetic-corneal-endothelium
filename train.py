"""
Training GAN

@author: Adrian Kucharski
"""
import os
from model import GAN
from pathlib import Path
from dataset import load_alizarine_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    gan = GAN(patch_per_image=500)
    # gan.summary()
    datasetAlizarine = load_alizarine_dataset(
         'datasets/fold_1/', mask_dilation=None)
    gan.train(50, datasetAlizarine, save_per_epochs=1)