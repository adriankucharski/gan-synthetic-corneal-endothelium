import shutil
import os

if __name__ == '__main__':
    paths = [
        'segmentation/logs/',
        'segmentation/models/raw/',
        'segmentation/params/raw/'
    ]
    for path in paths:
        shutil.rmtree(path)
        os.mkdir(path)
    