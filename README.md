# Corneal endothelium image segmentation using CNN trained with Patch-GAN generated data

Mosaic Generator (`hexgrid.py/grid_create_hexagons`) is based on https://github.com/MichaelMure/gimp-plugins/blob/master/common/mosaic.c

Patch-Based-UNet prediction is based on https://github.com/afabijanska/CornealEndothelium

Repository contains implementation of convolutional neural network models: Patch-GAN and Segmentation Sliding-Window UNet.

## Prerequisites
Code was tested on Windows 10 64-bit with Python 3.9.6, and TensorFlow 2.9.1. Befeore use, run `pip install -r requirements.txt` or `python -m pip install -r requirements.txt`.

## Config files

### gan_patch_generator_config.json
<pre><code>
{
  <b>"generator_path"</b> - Path to a trained generator
  <b>"path_to_save"</b> - Path to save generated mask-image pair
  <b>"hexagon_generator_params"</b> - Hexagons parameters
  {
    <b>"num_of_data"</b> - The number of generated images (int)
    <b>"batch_size"</b> - Batch size (int)
    <b>"hexagon_size"</b> - The pair of two numbers (default [18, 21]). Determinate the height of a generated hexagon (a corneal endothelium  cell)
    <b>"sap_ratio"</b> - The pair of two numbers (default [0.0, 0.1]) from 0 to 1. Add a salt (edges-like pixels) to generated mosaic hexagon images before passing it to a trained generator. It adds noise to data ([0.0, 0.1] means - make up to 10% of pixels be edges-like) and makes generated images more realistic.
    <b>"neatness_range"</b> [0.6, 0.8],
    <b>"sap_value_range"</b>: [0.2, 0.8],
    <b>"keep_edges"</b>: 0.8,
    <b>"remove_edges_ratio"</b>: 0.05,
    <b>"rotation_range"</b>: [-60, 60],
    <b>"inv_values"</b>: true
  }
}
</pre></code>