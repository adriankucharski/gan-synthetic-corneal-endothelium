import json
import sys
from skimage import io
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from util import add_salt_and_pepper
from typing import Tuple
from predict import UnetPrediction

class GANsPrediction(UnetPrediction):
    def __init__(self, model_path: str, patch_size: int = 64, stride: int = 4, batch_size: int = 64, salt=False):
        self.model: Model = load_model(model_path)
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.salt = salt
    
    def _predict_from_array(self, data: Tuple[np.ndarray], verbose=0) -> Tuple[np.ndarray]:
        predicted = []
        for idx in range(len(data)):
            height, width = data[idx].shape[:2]
            img = self._add_outline(data[idx])
            new_height, new_width = img.shape[:2]
            patches = 1 - self._get_patches(img)
            if self.salt:
                patches = add_salt_and_pepper(patches)
            noise = np.random.normal(size=patches.shape)
            
            predictions = self.model.predict([patches, noise], batch_size=self.batch_size, verbose=verbose)
            pred_img = self._build_img_from_patches(
                predictions, new_height, new_width)

            pred_img = pred_img[:height, :width]
            predicted.append(pred_img)
        return predicted

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 4:
        print('Provide at least four arguments')
        exit()
    datasets_names = ['Alizarine', 'Gavet', 'Hard', 'Rotterdam', 'Rotterdam_1000']
    
    dataset_path, fold, model_path, stride = args[0:4]

    gan = GANsPrediction(model_path, stride=int(stride))
    
    dataset_path = f'datasets/{dataset_path}/folds.json'
    with open(dataset_path, "r") as f:
        folds_json = json.load(f)
        
    if 'raw' in model_path:
        prefix = 'raw'
    else:
        prefix = 'synthetic'
    
    result_save = Path('./result_image_gan', prefix, folds_json['dataset_name'])
    print(str(result_save))
    result_save.mkdir(parents=True, exist_ok=True)
    
    parent_path = folds_json['dataset_path']
    for image in folds_json['folds'][int(fold)]['test']:
        pr = str(Path(parent_path, 'gt', image))
        mr = str(Path(parent_path, 'roi', image))
        ps = str(Path(result_save, image))
        pred = gan.predict(pr)[0]
        
        mask = io.imread(mr, as_gray=True).reshape(pred.shape) / 255
        io.imsave(ps, (pred + 1) / 2 * mask)