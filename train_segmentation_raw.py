"""
Train Segmentation UNet with raw data

@author: Adrian Kucharski
"""
import os

from dataset import images_preprocessing, load_dataset, DataIterator
from model import SegmentationUnet
from util import dumb_params
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":
    with open("config.json") as config_file:
        config = json.load(config_file)["unet.raw.training"]

    dataset_meta = config["dataset_meta"]
    preprocesing = config["preprocesing"]

    train, test = load_dataset(
        json_path=dataset_meta["path"],
        normalize=False,
        as_numpy=dataset_meta["as_numpy"],
    )[dataset_meta["fold"]]

    validation_data = DataIterator(
        test, 1, patch_per_image=1, inv_values=False
    ).get_dataset()

    masks, images = DataIterator(
        train, 1, patch_per_image=dataset_meta["patch_per_image"], inv_values=False
    ).get_dataset()

    dataset = images_preprocessing(
        images,
        masks,
        gamma_range=preprocesing["gamma_range"],
        noise_range=preprocesing["noise_range"],
        rotate90=preprocesing["rotate90"],
        gaussian_sigma=preprocesing["gaussian_sigma"],
    )
  
    unet = SegmentationUnet(
        log_path_save="segmentation/logs", model_path_save="segmentation/models/raw"
    )
    dumb_params(config, "segmentation/params/raw")
    unet.train(25, dataset, validation_data, validation_split=0.10)
    # os.system("shutdown /s /t 60")
