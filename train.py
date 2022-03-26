"""
Training GAN

@author: Adrian Kucharski
"""
import os
from model import GAN
from pathlib import Path
from dataset import load_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    fold = 0
    gan = GAN(patch_per_image=500)
    train, test  = load_dataset(r'datasets\Alizarine\folds.json')[fold]
    gan.train(50, train, save_per_epochs=1)