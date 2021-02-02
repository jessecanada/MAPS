# MAPS: automating Morphology And Phenotype Scoring
### Most cell biology experiments involve scoring the subcellular localizations of proteins or inspecting the morphologies of organelles from microscopy images. This process is very labour-intensive as it is traditionally done manually.
### MAPS is designed to automate the processing and interpretations of large amounts of microscopy images, but is adaptable to other types of images.
### MAPS is built using a combination of computer vision and machine learning (deep learning) techniques.
### Detailed implementation guide is published to protocols.io: dx.doi.org/10.17504/protocols.io.bn7dmhi6
###
#### Steps:
#### **Beofre we start:** Input images should be converted to TIFF files of microscopy images containing Red, Green and Blue (DAPI stained nuclei) channels.
#### **1. Pre-processing:** This is a quality control step to remove blurry (out-of-focus images). Also some basic histogram corrrection is applied.
#### **2.1 Cell detection:** An object detection model built on Azure Custom Vision is used to detect individual cells from images. Bounding boxes for detected cells are visualized. Each cell is cropped from the images.
#### **2.2. Training augmentation:** Augment your training data with 14 different image transformations. The original bounding box coordinates are preserved. Images are re-uploaded to Azure for re-training. Prereq: bbox_util.py
#### 3. **Data visualization:** Generate cell galleries to help you visualize phenotypes. First, image features are extracted by parallel stacking of 3 convolutional layers, then t-SNE (with PCA initiallization, Kobak & Berens Nat. Comm. 2019) and spectral clustering are applied.
#### **4. Phenotype scoring:** A classification model built on Azure is used to classify the cells into different bins, based on the general classes of localizations discovered in step 2.2.
### Note: Because different experiments involve different cell types and proteins, users should build their own object detection and classification models on Azure.
####  Visit this [quick start guide](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/get-started-build-detector) on how to build your first object detection model (no coding required! yay!).
####  Read this [how-to](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/python-tutorial) to learn about using your Azure Custom Vision model with the Python SDK.
