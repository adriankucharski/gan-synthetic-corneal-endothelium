# Corneal endothelium image segmentation using CNN trained with Patch-GAN generated data

Mosaic Generator (`hexgrid.py/grid_create_hexagons`) is based on https://github.com/MichaelMure/gimp-plugins/blob/master/common/mosaic.c

Patch-Based-UNet prediction is based on https://github.com/afabijanska/CornealEndothelium

Repository contains implementation of convolutional neural network models: Patch-GAN and Segmentation Sliding-Window UNet.

## Prerequisites
Code was tested on Windows 10 64-bit with Python 3.10.5, and TensorFlow 2.9.1. Befeore use, run `pip install -r requirements.txt` or `python -m pip install -r requirements.txt`.


## Streamlit App
To open a Streamlit app from a script named "main.py", you can use the command streamlit run main.py in your terminal or command prompt. Make sure that you have streamlit package installed on your machine before running this command.
