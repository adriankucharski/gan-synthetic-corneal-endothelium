import os
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if __name__ == '__main__':
    im_name = 'org_1.png'
    path = r'data\images\20220326-0132\*'

    frames = []
    for dir_name in sorted(glob(path), key=lambda x: int(Path(x).name)):
        epoch = Path(dir_name).name
        im_path = os.path.join(dir_name, im_name)
        img = Image.open(im_path)
        img = np.array(img)
        img = Image.fromarray(np.pad(img, 16))

        drawer = ImageDraw.Draw(img)
        drawer.text((4, 4), f'Epoch {epoch}', fill=255)
        frames.append(img)
    
    frame = frames.pop(0)
    frame.save(fp='gif.gif', format='GIF', append_images=frames,
         save_all=True, duration=300, loop=1)
