# FakeImageDetector_Model

In recent years, AI-generated images have become a powerful asset for the creative community, such as illustrators and designers, to assist their workflow. However, this same technology can also be misused by individuals with harmful intentions, who might leverage its capabilities to craft deceptive content.
The idea is to develop a pipeline to understand if a picture is real or generated, and if so, which kind of model generated it.
The proposed pipeline feeds data from the frequency spectrum rather than the original RGB image to a model which has to be able to distinguish between real and fake images, and in the latter case discriminate which generator was used in the creation of the original image itself.

## Requirements
* Python 3.10+
* Packages in the `requirements.txt` file

## Usage
To run the project it is possible to:
1. Set up parameters in the `parameters.ini` configuration file
2. Create the dataset using the `create_dataset.py` script
3. Preprocess the dataset using either the `preprocessing_no_denoise.py` script or the `preprocessing_denoise.py` script
4. Run the Cross Validation using the `cross_validation.py` script
