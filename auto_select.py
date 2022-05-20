import os
from glob import glob
from typing import Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from skimage import  io
from skimage.metrics import structural_similarity, normalized_mutual_information
from tqdm import tqdm
import skimage.filters as filters
import cv2

def get_best_k_generators_paths(path: str, k = 20, w = 64, show_bar=True, skip=0, paths_only=False) -> Union[Tuple[Tuple[str, int, float]], Tuple[str]]:
    mean_metrics = []
    results_path = sorted(glob(path), key=lambda x: int(os.path.split(x)[-1]))
    for idx, _dir in enumerate(tqdm(results_path)) if show_bar else enumerate(results_path):
        _metrics = []
        
        if idx < skip:
            continue
        
 
 
        
        for impath in glob(os.path.join(_dir, '*'))[:50]:
            im = io.imread(impath, as_gray=True) / 255
            y_pred, y_true = im[..., w:w*2], im[..., w*2:]

            a, _ = np.histogram(y_pred, 100, (0, 1))
            b, _ = np.histogram(y_true, 100, (0, 1))
            
            # a = cv2.calcHist([y_pred], [0], None, [100], [0, 1], accumulate=False)
            # b = cv2.calcHist([y_true], [0], None, [100], [0, 1], accumulate=False)
            
            p = cv2.compareHist(a.ravel().astype('float32'), b.ravel().astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
            
            # p, _ = pearsonr(a, b)
            s = structural_similarity(y_true, y_pred, data_range=1, win_size=15) 
            m = normalized_mutual_information(y_true, y_pred)
            # print(_dir, m, p)
            # input()
            # score = p + m + s
            score = (1/p) * 0.3 + s * 0.3 + m * 0.3
            _metrics.append(score)

        mean_metrics.append([_dir, idx, np.mean(_metrics)])
        
    mean_metrics.sort(key=lambda x: x[-1], reverse=True)
    return [p[0] for p in mean_metrics[:k]] if paths_only else mean_metrics[:k]
    
def get_best_of_the_bests(_path: str, k = 5, w = 64, show_bar=False, skip=0, paths_only=False) -> Union[Tuple[Tuple[str, int, float]], Tuple[str]]:
    bests = []
    for path in sorted(glob(os.path.join(_path, '*')), key=lambda x: int(os.path.split(x)[-1])):
        path = os.path.join(path, '**/*')
        _best = get_best_k_generators_paths(path, 1, w, show_bar, skip, paths_only)
        bests.append(_best[0])
        # print(_best[0])
    bests.sort(key=lambda x: x[-1])
    return bests[::-1][:k]
    
if __name__ == '__main__':
    for el in get_best_of_the_bests('data/images/20220503-2026/', skip=40, paths_only=True, k=10):
        print(el)
    # exit()
    # for i, path in enumerate(sorted(glob('data/images/20220503-2026/*'), key=lambda x: int(os.path.split(x)[-1]))):
    #     path = os.path.join(path, '**/*')
    #     bests = get_best_k_generators_paths(path, k = 1, w = 64, show_bar=False, skip=25)
    #     print(bests)
    
