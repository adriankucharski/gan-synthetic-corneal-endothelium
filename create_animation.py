from glob import glob
from PIL import Image
import os

if __name__ == '__main__':
    im_name = '1.png'
    path = 'data/images/20220322-2026/*'

    frames = []
    for dir_name in glob(path):
        im_path = os.path.join(dir_name, im_name)
        Image.open()