"""
Training Segmentation Unet

@author: Adrian Kucharski
"""
import os
from model import SegmentationUnet
from dataset import load_dataset, DataIterator
from predict import generate_dataset
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from util import add_marker_to_mask

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    synthetic_dataset = generate_dataset(r'generator\models\20220330-1516\model_last.h5', num_of_data=128 * 100, batch_size=256)
    image, mask = synthetic_dataset
    for i in range(len(image)):
        rgamma = np.random.uniform(0.25, 1.75)
        image[i] = exposure.adjust_gamma(image[i], rgamma)
        add_marker_to_mask(mask[i])
    synthetic_dataset = (image, mask)
    print(image.min(), image.max(), mask.min(), mask.max())

    
    unet = SegmentationUnet()
    unet.summary()

    fold = 0
    train, test  = load_dataset(r'datasets\Alizarine\folds.json', normalize=False)[fold]
    # image, mask = DataIterator(train, 1, inv_values=False).get_dataset()
    validation_data = DataIterator(test, 1, patch_per_image=1, inv_values=False).get_dataset()
    unet.train(50, (image, mask), validation_data, validation_split=0.1)