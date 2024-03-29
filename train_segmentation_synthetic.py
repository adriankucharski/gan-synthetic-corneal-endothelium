"""
Train Segmentation Unet with synthetic data

@author: Adrian Kucharski
"""
import json
import os
from typing import Tuple

import numpy as np
from dataset import (
    crop_patch,
    generate_dataset_from_generators,
    images_preprocessing,
    load_dataset,
    DataIterator,
)
from model import SegmentationUnet
from util import dumb_params

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":
    with open("config.json") as config_file:
        config = json.load(config_file)["unet.synthetic.training"]

    epochs = int(config["epochs"])
    validation_split = config["validation_split"]
    generators = config["generators"]
    dataset_meta = config["dataset_meta"]
    preprocesing = config["preprocesing"]
    hexagon_generator_params = config["hexagon_generator_params"]

    synthetic_masks, synthetic_images = generate_dataset_from_generators(
        generators, hexagon_generator_params
    )

    dataset = images_preprocessing(
        synthetic_images,
        synthetic_masks,
        gamma_range=preprocesing["gamma_range"],
        noise_range=preprocesing["noise_range"],
        rotate90=preprocesing["rotate90"],
        gaussian_sigma=preprocesing["gaussian_sigma"],
        corruption_range=preprocesing["corruption_range"],
        log_range=preprocesing["log_range"],
        standardization=preprocesing["standardization"]
    )

    _, test = load_dataset(
        dataset_meta["path"], normalize=False, as_numpy=dataset_meta["as_numpy"]
    )[dataset_meta["fold"]]
    validation_data = DataIterator(
        test, 1, patch_per_image=1, inv_values=False
    ).get_dataset()

    unet = SegmentationUnet(
        log_path_save="segmentation/logs",
        model_path_save="segmentation/models/synthetic",
    )
    dumb_params(config, "segmentation/params/synthetic")
    unet.train(epochs, dataset, validation_data, validation_split=validation_split)
