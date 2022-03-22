"""
Training GAN

@author: Adrian Kucharski
"""
import os
import pickle
from model import GAN
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    shape = 64
    epochs = 500
    batch_size = 128
    dataset = Path('./datasets/dataset.h5')
    dataset_validation = Path('./datasets/dataset_validation.h5')
    g_path_save =  Path('./models/generator')
    d_path_save = Path('./models/discriminator')


    with open(dataset, 'rb') as f:
        dataset = pickle.load(f)

    with open(dataset_validation, 'rb') as f:
        data_validation = pickle.load(f)

    

    # gan = GAN(shape=shape, g_path_save=str(g_path_save), d_path_save=str(d_path_save))
    # # gan.train_generator(epochs=epochs,
    # #                     dataset=dataset,
    # #                     val_data=data_validation)

    # gan.train(
    #     epochs=epochs,
    #     dataset=dataset,
    #     val_data=data_validation,
    #     batch_size=batch_size,
    # )
