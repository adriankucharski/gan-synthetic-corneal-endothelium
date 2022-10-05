"""
Dataset Generator with GAN Generator model
@author: Adrian Kucharski
"""

import json
from skimage import io
from pathlib import Path
from dataset import generate_dataset_from_generators

if __name__ == "__main__":
    with open('config.json') as config_file:
        config = json.load(config_file)['gan.patch.generator']
    
    synthetic_masks, synthetic_images = generate_dataset_from_generators(
        [config['generator_path']], config['hexagon_generator_params']
    )
    
    path_to_save = Path(config['path_to_save']) 
    path_to_save.mkdir(parents=True, exist_ok=True)
    
    for (mask, image, index) in zip(synthetic_masks, synthetic_images, range(len(synthetic_masks))):
        io.imsave(str(path_to_save / f'{index}_mask.png'), (mask * 255).astype("uint8"))
        io.imsave(str(path_to_save / f'{index}_image.png'), (image * 255).astype("uint8"))
        