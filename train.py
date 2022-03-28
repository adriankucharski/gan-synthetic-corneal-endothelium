"""
Training GAN

@author: Adrian Kucharski
"""
import os
from model import GAN
from dataset import load_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    fold = 0
    gan = GAN(patch_per_image=500)
    # gan.d_model.load_weights(r'discriminator\models\20220327-2229\model_last.h5')
    # gan.g_model.load_weights(r'generator\models\20220327-2229\model_last.h5')
    train, test  = load_dataset(r'datasets\Alizarine\folds.json')[fold]
    gan.train(200, train, save_per_epochs=1)