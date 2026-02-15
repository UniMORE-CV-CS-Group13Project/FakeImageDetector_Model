import logging
import glob
import configparser
import torch
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from FakeImageDetector import support_functions

# Configuration Parser
config = configparser.ConfigParser()
config.read("parameters.ini")

# Logger initialization
logger = logging.getLogger(
    name="Preprocessing"
)
formatter = logging.Formatter(
    fmt="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M"
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.setLevel("INFO")
logger.addHandler(console_handler)

## Set up logging to file
support_functions.create_folder(f"{config['General']['base_folder']}/{config['General']['logs_folder']}")
logging.basicConfig(
    filename=f"{config['General']['base_folder']}/{config['General']['logs_folder']}/preprocessing.log",
    encoding="utf-8",
    level=logging.INFO
)

# Create output folders (if they do not exist)
support_functions.create_folder(f"{config['General']['base_folder']}/{config['Dataset']['processed_ds_folder']}")
support_functions.create_folder(f"{config['General']['base_folder']}/{config['Dataset']['processed_ds_folder']}/data")
support_functions.create_folder(f"{config['General']['base_folder']}/{config['Dataset']['processed_ds_folder']}/labels")

def process_image(image_bytes):
    # Preprocessing logic -> Denoise and Fourier Transform
    try:
        pil_image = Image.open(BytesIO(image_bytes))
        img = np.array(pil_image)
        
        # Greyscale transformation
        if img.ndim == 2:
            grey = img
        elif img.ndim == 3 and img.shape[2] == 3:
            grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.ndim == 3 and img.shape[2] == 4:
            grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            error = "Image is not with canonical dimensions, returning zero tensor with correct shape"
            logger.error(error)
            raise Exception(error)
        
        # Resize the image to a standard size
        resized_img = cv2.resize(grey, (int(config['Model']['image_width']), int(config['Model']['image_height'])))
        
        # Apply denoising to the image
        denoised_img = cv2.fastNlMeansDenoising(
            src=resized_img,
            dst=None,
            h=3.0,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # FFT
        image_dft = np.fft.fft2(denoised_img)
        image_dft_shifted = np.fft.fftshift(image_dft)
        # Magnitude
        magnitude_spectrum = 20 * np.log10(np.maximum(np.abs(image_dft_shifted), 1e-8))
        # Phase
        phase_spectrum = np.angle(image_dft_shifted)
        sin_phase = np.sin(phase_spectrum)
        cos_phase = np.cos(phase_spectrum)
        # Autocorrelation
        image_dft_conj = np.conjugate(image_dft)
        power_density = image_dft * image_dft_conj
        autocorrelation = np.fft.fftshift(np.fft.ifft2(power_density).real)
        
        # Stack and transform from float64 to float32 to reduce memory footprint
        processed_image = np.stack([magnitude_spectrum, sin_phase, cos_phase, autocorrelation], axis=-1)
        return processed_image.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Unable to process image, returning zero tensor with correct shape. Error:\n{e}")
        return np.zeros((int(config['Model']['image_width']), int(config['Model']['image_height']), 4), dtype=np.float32)
    
if __name__ == "__main__":
    # Map: column_name -> label_id
    columns_map = {
        'image_gen0': 0,
        'image_gen1': 1,
        'image_gen2': 2,
        'image_gen3': 3,
        'image_real': 4
    }
    # Get list of available parquet files
    parquet_files = sorted(glob.glob(f"{config['General']['base_folder']}/{config['Dataset']['parquet_folder']}/*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files.")
    # Process each parquet file and produce an output
    for p_file in tqdm(parquet_files, desc="Processing Parquet Files", total=len(parquet_files)):
        logger.info(f"Opening file {p_file}")
        parquet = pd.read_parquet(p_file)
        logger.info(f"File {p_file} opened")
        transformed_imgs_chunk = []
        labels_chunk = []
        logger.info(f"File {p_file} - Processing images")
        for _, row in parquet.iterrows():
            for column_name, image in row.items():
                label = columns_map[column_name]
                processed_image = process_image(image)
                transformed_imgs_chunk.append(processed_image)
                labels_chunk.append(label)
        logger.info(f"File {p_file} - All images processed")
        transformed_img_chunks_np = np.array(transformed_imgs_chunk, dtype=np.float32)
        transformed_img_chunks_tensor = torch.from_numpy(transformed_img_chunks_np).permute(0, 3, 1, 2)
        torch.save(
            transformed_img_chunks_tensor,
            f"{config['General']['base_folder']}/{config['Dataset']['processed_ds_folder']}/data/{p_file.split('/')[-1].split('.')[0]}.pt"
        )
        logger.info(f"File {p_file} - Image Tensor saved")
        labels_chunk_np = np.array(labels_chunk, dtype=np.int8)
        labels_chunk_tensor = torch.from_numpy(labels_chunk_np)
        torch.save(
            labels_chunk_tensor,
            f"{config['General']['base_folder']}/{config['Dataset']['processed_ds_folder']}/labels/{p_file.split('/')[-1].split('.')[0]}.pt"
        )
        logger.info(f"File {p_file} - Label Tensor saved")