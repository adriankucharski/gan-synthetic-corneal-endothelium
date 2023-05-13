# Corneal endothelium image segmentation using CNN trained with Patch-GAN generated data

# Terms of use
<p>The source code and generated image dataset may be used for non-commercial research, provided you acknowledge the source by citing the following paper:<p>
<ul>
  <li> Adrian Kucharski, Anna Fabijańska, Corneal endothelial image segmentation training data generation using GANs. Do experts need to annotate?, Biomedical Signal Processing and Control, Volume 85, 2023, 104985, ISSN 1746-8094, https://doi.org/10.1016/j.bspc.2023.104985. (https://www.sciencedirect.com/science/article/pii/S1746809423004184)</li>
</ul>

<pre><code>@article{KUCHARSKI2023104985,
title = {Corneal endothelial image segmentation training data generation using GANs. Do experts need to annotate?},
journal = {Biomedical Signal Processing and Control},
volume = {85},
pages = {104985},
year = {2023},
issn = {1746-8094},
doi = {https://doi.org/10.1016/j.bspc.2023.104985},
url = {https://www.sciencedirect.com/science/article/pii/S1746809423004184},
author = {Adrian Kucharski and Anna Fabijańska},
keywords = {Corneal endothelium, Synthetic train data, Generative adversarial neural networks, Image segmentation, U-Net},
abstract = {Objective.
Current solutions for corneal endothelial image segmentation use convolutional neural networks. However, their potential is not exploited due to the scarcity of labeled corneal endothelial data caused by an expensive cell delineation process. Therefore, this work proposes synthesizing cell edges and corresponding images using generative adversarial neural networks. To our knowledge, such work has not yet been reported in the context of corneal endothelial image segmentation.
Methods.
A two-step pipeline for synthesizing training patches is proposed. First, a custom mosaic generator creates a grid that mimics a binary map of endothelial cell edges. The synthetic cell edges are then input to the generative adversarial neural network, which is trained to generate corneal endothelial image patches from the corresponding edge labels. Finally, pairs of synthetic patches are used to train the patch-based U-Net model for corneal endothelial image segmentation.
Results.
Experiments performed on three datasets of corneal endothelial images show that using purely synthetic data for U-Net training allows image segmentation with comparable accuracy to that obtained when using original images annotated by experts. Depending on the dataset, the segmentation quality decreases only from 1% to 4%.
Conclusions:
Our solution provides a cost-effective source of diverse training data for corneal endothelial image segmentation.
Significance.
Due to the simple graphical user interface wrapping the proposed solution, many users can easily obtain unlimited training data for corneal endothelial image segmentation and use it in various scenarios.}
}</code></pre>

Mosaic Generator (`hexgrid.py/grid_create_hexagons`) is based on https://github.com/MichaelMure/gimp-plugins/blob/master/common/mosaic.c

Patch-Based-UNet prediction is based on https://github.com/afabijanska/CornealEndothelium

Repository contains implementation of convolutional neural network models: Patch-GAN and Segmentation Sliding-Window UNet.

## Prerequisites
Code was tested on Windows 10 64-bit with Python 3.10.9, and TensorFlow 2.9.1. Befeore use, run `pip install -r requirements.txt` or `python -m pip install -r requirements.txt`.


## Streamlit App
To run a Streamlit app you can use the command `streamlit run main.py` or `python -m streamlit run main.py` in your terminal or command prompt. Make sure that you have streamlit package installed on your machine before running this command.
If you get `ImportError: cannot import name 'builder' from 'google.protobuf.internal'` run `python -m pip install --upgrade protobuf` or `python -m pip install protobuf==3.20.0`.

You can generate and download a corneal endothelium images dataset with a GUI. You can also use the `gan_patch_generator.ipynb` notebook to do so.

![plot](./docs/app_frontend.png)