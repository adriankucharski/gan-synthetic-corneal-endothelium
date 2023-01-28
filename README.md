# Corneal endothelium image segmentation using CNN trained with Patch-GAN generated data

Mosaic Generator (`hexgrid.py/grid_create_hexagons`) is based on https://github.com/MichaelMure/gimp-plugins/blob/master/common/mosaic.c

Patch-Based-UNet prediction is based on https://github.com/afabijanska/CornealEndothelium

Repository contains implementation of convolutional neural network models: Patch-GAN and Segmentation Sliding-Window UNet.

## Prerequisites
Code was tested on Windows 10 64-bit with Python 3.9.6, and TensorFlow 2.9.1. Befeore use, run `pip install -r requirements.txt` or `python -m pip install -r requirements.txt`.

## Config files

### config.json - gan.patch.generator
<pre><code>
{
  <b>"generator_path"</b> - Path to a trained generator
  <b>"path_to_save"</b> - Path to save generated mask-image pairs
  <b>"hexagon_generator_params"</b> - Hexagons parameters
  {
    <b>"num_of_data"</b> - The number of generated images
    <b>"batch_size"</b> - Batch size
    <b>"hexagon_height"</b> - A pair of two numbers (default [18, 21]). 
        This parameter determines the height of a generated hexagon (a corneal endothelium cell).
    <b>"neatness_range"</b> A pair of two numbers (default [0.6, 0.8]) from 0 to 1. 
        This parameter determines the deformation of generated mosaic hexagons.
        A value 1 means a perfect hexagon.
    <b>"sap_ratio"</b> - A pair of two numbers (default [0.0, 0.1]) from 0 to 1. 
        Add a salt (with a value from <b>sap_value_range</b>) to generated mosaic hexagon images before passing it to a trained generator. 
        It adds noise to data ([0.0, 0.1] means - make up to 10% of pixels to have a value from <b>sap_value_range</b>) and makes generated images more realistic.
    <b>"sap_value_range"</b> - A pair of two numbers (default [0.2, 0.8]) from 0 to 1.
        Related to the <b>"sap_ratio"</b>.
    <b>"keep_edges"</b> - The number from 0 to 1 (default 0.8). 
        This parameter determines how many (in %) pixels of edges won't be affected by the salt.
    <b>"remove_edges_ratio"</b> - The number from 0 to 1 (default 0.05).
        This parameter determines how many edges will be removed from generated mosaic hexagons. 
        A value of 0 means that no edges will be removed.
    <b>"rotation_range"</b> - A pair of two numbers (default [-60, 60]).
        A rotation in degrees of a generated mosaic.
    <b>"inv_values"</b> - True or false (default true). 
        If true: edges are white (1), cells bodies are black (0). 
        If false: edges are black (0), cells bodies are white (1).
  }
}
</code></pre>

### config.json
<pre><code>
