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
        config = json.load(config_file)["unet.mix.training"]

    # Gavet
    # _generators = [
    #     [
    #         "generator\\models\\20220429-0021\\model_71.h5",
    #         "generator\\models\\20220429-0021\\model_76.h5",
    #         "generator\\models\\20220429-0021\\model_81.h5",
    #         "generator\\models\\20220429-0021\\model_88.h5",
    #     ],
    #     [
    #         "generator\\models\\20220609-2213\\model_68.h5",
    #         "generator\\models\\20220609-2213\\model_76.h5",
    #         "generator\\models\\20220609-2213\\model_97.h5",
    #         "generator\\models\\20220609-2213\\model_99.h5",
    #     ],
    #     [
    #         "generator\\models\\20220609-2319\\model_65.h5",
    #         "generator\\models\\20220609-2319\\model_83.h5",
    #         "generator\\models\\20220609-2319\\model_94.h5",
    #         "generator\\models\\20220609-2319\\model_97.h5",
    #     ],
    # ]

    # Alizarine
    _generators = [
        [
            "generator\\models\\20220405-2359\\model_144.h5",
            "generator\\models\\20220405-2359\\model_145.h5",
            "generator\\models\\20220405-2359\\model_146.h5",
            "generator\\models\\20220405-2359\\model_147.h5",
        ],
        [
            "generator\\models\\20220603-1720\\model_114.h5",
            "generator\\models\\20220603-1720\\model_120.h5",
            "generator\\models\\20220603-1720\\model_141.h5",
            "generator\\models\\20220603-1720\\model_143.h5",
        ],
        [
            "generator\\models\\20220603-1850\\model_137.h5",
            "generator\\models\\20220603-1850\\model_141.h5",
            "generator\\models\\20220603-1850\\model_145.h5",
            "generator\\models\\20220603-1850\\model_147.h5",
        ],
    ]
    # Rotterdam
    # _generators = [
    #     [
    #         "generator/models/20220623-0916/model_15.h5",
    #         "generator/models/20220623-0916/model_16.h5",
    #         "generator/models/20220623-0916/model_20.h5",
    #         "generator/models/20220623-0916/model_68.h5",
    #     ],
    #     [
    #         "generator\\models\\20220609-1841\\model_15.h5",
    #         "generator\\models\\20220609-1841\\model_16.h5",
    #         "generator\\models\\20220609-1841\\model_17.h5",
    #         "generator\\models\\20220609-1841\\model_22.h5",
    #     ],
    #     [
    #         "generator\\models\\20220609-2031\\model_34.h5",
    #         "generator\\models\\20220609-2031\\model_37.h5",
    #         "generator\\models\\20220609-2031\\model_72.h5",
    #         "generator\\models\\20220609-2031\\model_83.h5",
    #     ],
    # ]

    for i in range(3):
        config["generators"] = _generators[i]
        config["dataset_meta"]["fold"] = i

        epochs = int(config["epochs"])
        validation_split = config["validation_split"]

        generators = config["generators"]
        dataset_meta = config["dataset_meta"]
        preprocesing = config["preprocesing"]
        hexagon_generator_params = config["hexagon_generator_params"]

        train, test = load_dataset(
            json_path=dataset_meta["path"],
            normalize=False,
            as_numpy=dataset_meta["as_numpy"],
        )[dataset_meta["fold"]]

        raw_masks, raw_images = DataIterator(
            train, 1, patch_per_image=dataset_meta["patch_per_image"], inv_values=False
        ).get_dataset()

        if hexagon_generator_params["num_of_data"] == -1:
            hexagon_generator_params["num_of_data"] = len(raw_images) // len(generators)
        synthetic_masks, synthetic_images = generate_dataset_from_generators(
            generators, hexagon_generator_params
        )

        images = np.concatenate((synthetic_images, raw_images), axis=0)
        masks = np.concatenate((synthetic_masks, raw_masks))

        dataset = images_preprocessing(
            images,
            masks,
            gamma_range=preprocesing["gamma_range"],
            noise_range=preprocesing["noise_range"],
            rotate90=preprocesing["rotate90"],
            gaussian_sigma=preprocesing["gaussian_sigma"],
            corruption_range=preprocesing["corruption_range"],
            log_range=preprocesing["log_range"],
            standardization=preprocesing["standardization"],
        )

        validation_data = DataIterator(
            test, 1, patch_per_image=1, inv_values=False
        ).get_dataset()

        unet = SegmentationUnet(
            log_path_save="segmentation/logs",
            model_path_save="segmentation/models/mix",
        )
        dumb_params(config, "segmentation/params/mix")
        unet.train(epochs, dataset, validation_data, validation_split=validation_split)
