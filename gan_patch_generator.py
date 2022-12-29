"""
Dataset Generator with GAN Generator model
@author: Adrian Kucharski
"""

import json
from skimage import io, exposure
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from dataset import crop_patch, generate_dataset_from_generators, images_preprocessing

if __name__ == "__main__":
    with open("config.json") as config_file:
        config = json.load(config_file)["gan.patch.generator"]
        preprocesing = config["preprocesing"]
    synthetic_masks, synthetic_images = generate_dataset_from_generators(
        [config["generator_path"]], config["hexagon_generator_params"]
    )

    path_to_save = Path(config["path_to_save"])
    path_to_save.mkdir(parents=True, exist_ok=True)

    synthetic_images, synthetic_masks = images_preprocessing(
        synthetic_images,
        synthetic_masks,
        gamma_range=preprocesing["gamma_range"],
        noise_range=preprocesing["noise_range"],
        rotate90=preprocesing["rotate90"],
        gaussian_sigma=preprocesing["gaussian_sigma"],
        corruption_range=preprocesing["corruption_range"],
        log_range=preprocesing["log_range"],
        standardization=preprocesing["standardization"],
    )

    plt.imshow(np.hstack(synthetic_images), 'gray')
    plt.show()

    # for (mask, image, index) in zip(
    #     synthetic_masks, synthetic_images, range(len(synthetic_masks))
    # ):
    #     io.imsave(str(path_to_save / f"{index}_mask.png"), (mask * 255).astype("uint8"))
    #     io.imsave(
    #         str(path_to_save / f"{index}_image.png"), (image * 255).astype("uint8")
    #     )
