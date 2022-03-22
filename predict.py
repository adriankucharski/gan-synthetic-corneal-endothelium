from skimage import io
import os
from pathlib import Path
import numpy as np
from model import ModelPrediction
from util import imsread


if __name__ == '__main__':
    shape = 64
    model_path = Path('./models/generator/20220128-1958/model_last.h5')
    images_paths = Path('./data/tom_and_jerry/validation/jerry/')
    images_path_save = Path('./prediction/')



    model = ModelPrediction(str(model_path), shape=shape)
    images = imsread(str(images_paths), as_float=True)
    images_pred = model.predict(images, orginal_shape=True)

    images_path_save.mkdir(parents=True, exist_ok=True)
    for (index, pred) in enumerate(images_pred):
        pred = np.array(pred * 255.0, dtype='uint8')
        io.imsave(os.path.join(images_path_save, f'{index}.png'), pred)
        
