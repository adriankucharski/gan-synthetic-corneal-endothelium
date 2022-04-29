from skimage import io
from glob import glob
import os
import numpy as np
from skimage.metrics import structural_similarity
from skimage.filters import gaussian
from typing import Tuple
from tqdm import tqdm

def get_best_k_generators_paths(path: str, k = 20, w = 64) -> Tuple[Tuple[str], Tuple[int]]:
    mean_metrics = []
    results_path = sorted(glob(path), key=lambda x: int(os.path.split(x)[-1]))
    for _dir in tqdm(results_path):
        _metrics = []
        for impath in glob(os.path.join(_dir, '*')):
            im = io.imread(impath, as_gray=True)
            y_pred, y_true = im[..., w:w*2], im[..., w*2:]
            y_pred, y_true = gaussian(y_pred), gaussian(y_true)
            score = structural_similarity(y_true, y_pred, data_range=1.0)
            _metrics.append(score)
        mean_metrics.append(np.mean(_metrics))
    sorted_index = np.argsort(mean_metrics)[::-1][:k]
    return np.array(results_path)[sorted_index], sorted_index
        
if __name__ == '__main__':
    w = 64
    path = 'data/images/20220405-2359/*'

    paths, indexes = get_best_k_generators_paths(path, k = 20, w = 64)
    print(paths)
    print(indexes)
