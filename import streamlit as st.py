import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

st.write("Disease Prediction on image")

uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "png"])
if uploaded_file is not None:
    # Define the save path
    save_path = f"file"
    
    # Save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File saved successfully: {save_path}")
else:
    st.warning("Please upload a file to save.")

model=tf.keras.models.load_model("vgg16_Mri.keras")
model.summary()
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Load and resize image
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)  # Preprocess for VGG16
    return img_array

# Make a prediction
def predict_image(model, image_path, class_labels):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)  # Predict using the model
    predicted_class = np.argmax(predictions, axis=1)  # Get the class with the highest probability
    return class_labels[predicted_class[0]], predictions[0][predicted_class[0]]
# if uploaded_file is not None:
image_path = 'Data/test/COVID19/COVID19(461).jpg'  # Replace with your image path
class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA']  # Define your class labels

predicted_label, confidence = predict_image(model, image_path, class_labels)
# print(f"Predicted class: {predicted_label} with confidence {confidence:.2f}")
if uploaded_file is not None:
    st.write(f"Predicted class: {predicted_label} with confidence {confidence:.2f}")