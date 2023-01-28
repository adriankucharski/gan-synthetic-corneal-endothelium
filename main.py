import numpy as np
import shutil
import streamlit as st
from glob import glob
import tempfile
from pathlib import Path
from skimage import io
from dataset import generate_dataset_from_generators, images_preprocessing

@st.cache
def to_tempfile(images_a: np.ndarray, images_b: np.ndarray):
    temp_dir = tempfile.mkdtemp()
    images_save_path: Path = Path(temp_dir)
    images_save_path.mkdir(parents=True, exist_ok=True)

    for i, (image_a, image_b) in enumerate(zip(images_a, images_b)):
        path_mask = str(images_save_path / f'{i}_mask.png')
        path_image = str(images_save_path / f'{i}_image.png')
        io.imsave(path_mask, np.array(image_a * 255, dtype='uint8'))
        io.imsave(path_image, np.array(image_b * 255, dtype='uint8'))

    shutil.make_archive(str(images_save_path), 'zip', str(images_save_path))
    return str(images_save_path) + '.zip'

batch_size = 128
max_num_to_show = 100
generators = list(glob(str(Path('./extra_data/generators/**/*.h5'))))

if __name__ == '__main__':
    st.title('Corneal Endothelium images and masks generator')
    
    generator = st.selectbox('GAN generator path', generators, index=0)

    num_of_data_text = st.text_input('Number of data to generate', value=5, type='default')
    
    with st.expander('Mosaic generator parameters', expanded=False):
        hexagon_height = st.slider("Mask hexagon height", 4, 50, (17, 25))
        neatness_range = st.slider("Neatness", 0.2, 1.0, (0.75, 0.95))
        remove_edges_ratio = st.slider("Remove edges ratio", 0.00, 0.20, 0.05)
        rotation = st.slider("Rotation (degrees)", -60, 60, (-45, 45))
        sap_ratio = st.slider("Salt ratio", 0.00, 0.20, (0.01, 0.1))
        sap_value_range = st.slider("Salt value", 0.00, 1.00, (0.2, 0.8))
        
    with st.expander('Generated images postprocessing', expanded=False):
        gamma_enabled = st.checkbox('Gamma')
        if gamma_enabled:
            gamma_range = st.slider('Gamma range', 0.2, 2.0, (1.0, 1.2))
        else:
            gamma_range = None
            
        log_enabled = st.checkbox('Log')
        if log_enabled:
            log_range = st.slider('Log', 0.2, 2.0, (1.0, 1.0))
        else:
            log_range = None
            
        noise_enabled = st.checkbox('Noise')
        if noise_enabled:
            noise_range = st.slider('Noise range', -0.1, 0.1, (-0.01, 0.01))
        else:
            noise_range = None
            
        gaussian_enabled = st.checkbox('Gaussian sigma')
        if gaussian_enabled:
            gaussian_sigma = st.slider('Sigma', 0.5, 2.0, 1.0)
        else:
            gaussian_sigma = 1.0
            
        rotate90 = st.checkbox('Rotate 90 degrees')

    show_data = st.checkbox('Show generated data', True)
    generate_button = st.button('Generate data')
    do_postprocessing = gamma_enabled or log_enabled or gaussian_enabled or rotate90

    try:
        num_of_data = int(num_of_data_text)
    except:
        num_of_data = 5

    postprocessing = {
        "gamma_range": gamma_range,
        "log_range": log_range,
        "rotate90": rotate90,
        "noise_range": noise_range,
        "gaussian_sigma": gaussian_sigma,
        "corruption_range": None,
        "standardization": False
    }
    hexagon_generator_params = {
        "num_of_data": int(num_of_data_text),
        "hexagon_height": hexagon_height,
        "batch_size": batch_size,
        "neatness_range": neatness_range,
        "remove_edges_ratio": remove_edges_ratio,
        "rotation_range": rotation,
        "sap_ratio": sap_ratio,
        "sap_value_range": sap_value_range,
        "keep_edges": 0.8,
        "inv_values": True
    }

    if generate_button:
        synthetic_masks, synthetic_images = generate_dataset_from_generators([generator], hexagon_generator_params)

        if do_postprocessing:
            synthetic_images, synthetic_masks = images_preprocessing(
                synthetic_images,
                synthetic_masks,
                **postprocessing
            )
        zip_path = to_tempfile(synthetic_masks, synthetic_images)

        st.success('Dataset generated successfully', icon="✅")
        with open(zip_path, "rb") as f:
            download = st.download_button(
                label="Download",
                data=f,
                file_name="generated_images.zip",
                mime="application/zip"
            )
        
        if show_data:
            images = []
            for i in range(min(num_of_data, max_num_to_show)):
                images.append(np.hstack([synthetic_masks[i], synthetic_images[i]]))
            images = np.vstack(images)
            st.image(images)