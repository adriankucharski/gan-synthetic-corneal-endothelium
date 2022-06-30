from glob import glob
import shutil
import os
from pathlib import Path

if __name__ == '__main__':
    exclude = [
        '20220602-1258',
        '20220602-1413',
        '20220603-0826',
        '20220625-0058',
        '20220628-0026',
        '20220625-0118',
        '20220610-0051',
        '20220602-2030',
        '20220610-0031',
        '20220625-0105',
        '20220628-1037',
        '20220625-0124',
        '20220610-0102',
        '20220610-0021',
        '20220610-0038',
        '20220625-0111',
        '20220628-1043',
        '20220625-0131',
        '20220628-1221'
    ]
    print(len(exclude))
    paths = [
        'segmentation/logs/*',
        'segmentation/models/raw/*',
        'segmentation/models/synthetic/*',
        'segmentation/params/raw/*',
        'segmentation/params/synthetic/*'
    ]
    for path in paths:
        for segpath in glob(path):
            if not any([ex in segpath for ex in exclude]):
                try:
                    shutil.rmtree(segpath)
                except:
                    os.remove(segpath)
        # shutil.rmtree(path)
        # os.mkdir(path)
