# MAPS: automating Morphology And Phenotype Scoring
### Most cell biology experiments involve scoring the subcellular localizations of proteins or inspecting the morphologies of organelles from microscopy images. This process is very labour-intensive as it is traditionally done manually.
### MAPS is designed to automate the processing and interpretations of large amounts of microscopy images, but is adaptable to other types of images.
### MAPS is built with a combination of computer vision and machine learning (deep learning) techniques.
### It is written in python 3.x. After importing images, all preprocessing steps are written with openCV. Post-processing is written with Sci-kit Learn. Deep learning models are deployed on Azure Custom Vision. Pandas handles all data.
### This project is a work in progress, and will be updated regularly.
###
#### Steps:
#### 1. Input images: TIFF files of microscopy images containing Red, Green and Blue (DAPI stained nuclei) channels.
#### 1.1. Pre-processing: This is a quality control step to remove blurry (out-of-focus images). Also some basic histogram corrrection is applied.
#### 2. Cell detection: An object detection model built on Azure Custom Vision is used to detect individual cells from images. Bounding boxes for detected cells are visualized. Each cell is cropped from the images.
#### 2.1. Training augmentation: The functions here help you augment your training data, while preserving the bounding box coordinates of the original training images. Pre-req: bbox_util.py
#### 2.2. Post-processing: The visualization functions implemented here will help you inspect your data space, allowing you to get a general sense of the morphologies of the cropped cells and identify sub-types. First, image features are extracted by parallel stacking of 3 convolutional layers, then PCA, t-SNE and spectral clustering are applied.
#### 3. Phenotype scoring: A classification model built on Azure is used to classify the cells into different bins, based on the general classes of localizations discovered in step 2.2.
### Note: Because different experiments involve different types of cells and proteins being analyzed, users should build their own object detection and classification models on Azure.
####  Visit this [quick start guide](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/get-started-build-detector) on how to build your first object detection model (no coding required! yay!).
####  Read this [how-to](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/python-tutorial) to learn about using your Azure Custom Vision model with the Python SDK.
