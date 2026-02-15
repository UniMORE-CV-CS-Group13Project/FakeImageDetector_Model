import os
import configparser
import logging
import requests
from tqdm import tqdm
from io import BytesIO
from PIL import Image
from PIL import UnidentifiedImageError
import pandas as pd
from datasets import load_dataset
from FakeImageDetector import support_functions

# Configuration Parser
config = configparser.ConfigParser()
config.read("parameters.ini")

# Logger initialization
logger = logging.getLogger(
    name="Dataset"
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
support_functions.create_folder(f"{config["General"]["base_folder"]}/{config["General"]["logs_folder"]}")
logging.basicConfig(
    filename=f"{config["General"]["base_folder"]}/{config["General"]["logs_folder"]}/dataset.log",
    encoding="utf-8",
    level=logging.INFO
)

def create_parquet(dataset, wanted_rows):
    chunk_counter = 0
    total_collected = 0

    # Parquet structure
    images = {'image_real': [], 'image_gen0': [], 'image_gen1': [], 'image_gen2': [], 'image_gen3': []}
    
    # Session initialization
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

    
    with tqdm(total=wanted_rows, desc="Downloading and storing images") as pbar:
        for row in dataset:
            # Check if wanted number of examples is reached
            if total_collected >= wanted_rows:
                break

            # Get real image url
            url = row.get('url')
            if url and support_functions.is_valid_url(url):
                try:
                    # Get real image from web
                    response = session.get(url, timeout=2)
                    response.raise_for_status()
                    image_content = response.content
                    pil_image = Image.open(BytesIO(image_content))

                    # Append real image to the correct column
                    images['image_real'].append(support_functions.image_to_bytes(pil_image))
                    # Gather the generated images from the Parquet
                    for i in range(4):
                        images[f'image_gen{i}'].append(support_functions.image_to_bytes(row.get(f'image_gen{i}')))

                    pbar.update(1)
                    total_collected += 1

                    # Chunk save
                    if len(images['image_real']) >= int(config["Dataset"]["chunk_size"]):
                        df_chunk = pd.DataFrame(images)

                        file_name = f"{config["General"]["base_folder"]}/{config["Dataset"]["parquet_folder"]}/dataset_part_{chunk_counter:03}.parquet"
                        df_chunk.to_parquet(file_name, engine='pyarrow')

                        logger.info(f"Saved chunk {chunk_counter} ({len(df_chunk)} rows) to {file_name}")

                        # Clear out RAM
                        images = {'image_real': [], 'image_gen0': [], 'image_gen1': [], 'image_gen2': [], 'image_gen3': []}
                        chunk_counter += 1

                except UnidentifiedImageError:
                    continue
                except Exception as e:
                    continue
    
    # Save last chunk if some images are still in the buffer
    if len(images['image_real']) > 0:
        df_chunk = pd.DataFrame(images)
        file_name = f"{config["General"]["base_folder"]}/{config["Dataset"]["parquet_folder"]}/dataset_part_{chunk_counter}.parquet"
        df_chunk.to_parquet(file_name, engine='pyarrow')
        logger.info(f"Saved final chunk {chunk_counter} ({len(df_chunk)} rows) to {file_name}")

    return total_collected

if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset("elsaEU/ELSA_D3", split='train', streaming=True)
    # Create folder to save outputs to
    support_functions.create_folder(f"{config["General"]["base_folder"]}/{config["Dataset"]["parquet_folder"]}")
    # Create Parquets
    downloaded_rows = create_parquet(dataset, int(config["Dataset"]["wanted_rows"]))
    logger.info(f"Download completed. Total rows saved: {downloaded_rows}")