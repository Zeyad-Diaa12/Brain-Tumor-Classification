# Brain Tumor Classification Using Convolutional Neural Networks (CNN)

This project aims to classify brain tumor images using a Convolutional Neural Network (CNN) model. The CNN model is trained on a dataset containing brain MRI scans with labeled tumor regions. The model can accurately classify images into one of the predefined classes: tumor present or tumor absent.

## Dataset
The dataset used for training and evaluation consists of brain MRI scans collected from various sources. Each MRI scan is labeled with binary classes indicating the presence or absence of a tumor. The dataset is split into training, validation, and test sets to train, validate, and evaluate the CNN model's performance.

## CNN Architecture
The CNN model used for brain tumor classification comprises several convolutional layers followed by max-pooling layers for feature extraction. Batch normalization and dropout layers are incorporated to prevent overfitting. The final layers consist of fully connected layers with softmax activation for classification.
