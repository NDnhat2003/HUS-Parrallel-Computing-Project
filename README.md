# HUS-Parrallel-Computing-Project
### Installation Guide: [Link](https://docs.google.com/document/d/1NBeOLqJkN8_gzrhpzCzn2G9lyDv0whV0397OPYAUVU4/edit#heading=h.nuccqh6shzfa)
### Model
- Preprocessing: The input CT Lung Scan image is first enhanced through Histogram Equalization to improve contrast, followed by Thresholding Segmentation to isolate objects from the background.
- Training Phase: Features are extracted from the preprocessed image using a Convolutional Neural Network (CNN), which is crucial for the model’s learning process.
- Testing Phase: The model evaluates the extracted features to classify the scan as either Cancerous or Non-cancerous.
### DataSet
The model uses a Lung Cancer CT Scan Image Dataset for training and evaluation processes. The Dataset consists of 1000 images which are classified in 4 folders according to 4 types of Lung cancer: Adenocarcinoma (338 images), Large Cell Carcinoma (187 images), Normal (215 images) and Squamous Cell Carcinoma (260 images). In preprocessing phrase, we standardized this dataset into 2 group: Cancerous and non-cancerous.
