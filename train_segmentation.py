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
from util import add_marker_to_mask, add_jpeg_compression
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    synthetic_dataset = generate_dataset(
        r'generator\models\20220330-1516\model_last.h5', 
        num_of_data=256 * 30,
        hexagon_size=(16, 21),
        batch_size=256,
        sap_ratio=(0.1, 0.5),
        normal_noise=(0, 1e-3),
        sap_value_range=(0.5, 1.0),
        result_gamma_range=(0.75, 1.75), 
        keep_edges=0.5,
    )
    synthetic_image, synthetic_mask = synthetic_dataset
    # for i in range(len(image)):
    #     q = np.random.randint(55, 100)
    #     image[i] = add_jpeg_compression(image[i], q)
    synthetic_dataset = (synthetic_image, synthetic_mask)

    print(synthetic_image.min(), synthetic_image.max(), synthetic_mask.min(), synthetic_mask.max())


    fold = 0
    train, test  = load_dataset(r'datasets\Alizarine\folds.json', normalize=False)[fold]
    mask, image = DataIterator(train, 1, inv_values=False).get_dataset()
    validation_data = DataIterator(test, 1, patch_per_image=1, inv_values=False).get_dataset()

    final_image = np.concatenate([synthetic_image, image], axis=0)
    final_mask = np.concatenate([synthetic_mask, mask], axis=0)


    unet = SegmentationUnet()
    unet.train(50, (final_image, final_mask), validation_data, validation_split=0.15)
    os.system('shutdown -s')