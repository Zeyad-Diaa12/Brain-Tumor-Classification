# Brain Tumor Classification Using Convolutional Neural Networks (CNN)

This project aims to classify brain tumor images using a Convolutional Neural Network (CNN) model. The CNN model is trained on a dataset containing brain MRI scans with labeled tumor regions. The model can accurately classify images into one of the predefined classes: notumor, glioma, meningioma and pituitary.

## Dataset
The dataset used for training and evaluation consists of brain MRI scans collected from various sources. Each MRI scan is labeled with class indicating notumor, glioma, meningioma or pituitary. The dataset is split into training, validation, and test sets.

## CNN Architecture
The CNN model used for brain tumor classification comprises 3 convolutional layers each of them is followed by max-pooling layer for feature extraction.Dropout layer are incorporated to prevent overfitting. The final layers consist of fully connected layers with softmax activation for classification.
