# Corneal endothelium image segmentation using CNN trained with Patch-GAN generated data

Mosaic Generator (`hexgrid.py/grid_create_hexagons`) is based on https://github.com/MichaelMure/gimp-plugins/blob/master/common/mosaic.c

Patch-Based-UNet prediction is based on https://github.com/afabijanska/CornealEndothelium

Repository contains implementation of convolutional neural network models: Patch-GAN and Segmentation Sliding-Window UNet.

## Prerequisites
Code was tested on Windows 10 64-bit with Python 3.9.6, and TensorFlow 2.9.1. Befeore use, run `pip install -r requirements.txt` or `python -m pip install -r requirements.txt`.

## Config files

### gan_patch_generator_config.json
We use it to