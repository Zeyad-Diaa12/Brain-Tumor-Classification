import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load pretrained model
model = tf.keras.models.load_model('F:\DL Projects\Brain Tumor Classification\Brain_Tumor_Model.h5')

# Define function to preprocess image
def preprocess_image(img):
    img = tf.image.resize(img, (224, 224))  
    img /= 255.0
    return img

def predict_image(image):
    img = preprocess_image(image)
    img = tf.expand_dims(img, axis=0)
    prediction = np.argmax(model.predict(img), axis = 1)
    return prediction


st.title('Brain MRI Image Prediction')

uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

classes = {0:'notumor',1:'glioma',2:'meningioma',3:'pituitary'}

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = predict_image(image)
    prediction_index = int(prediction) 
    st.write(f"Prediction: {classes.get(prediction_index)}")
