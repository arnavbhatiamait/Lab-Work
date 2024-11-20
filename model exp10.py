import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

# File upload for the model
model_file = st.file_uploader("Upload a trained model file", type=["h5"])

# File upload for the image
image_file = st.file_uploader("Upload an image file", type=["jpg", "png"])

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize image
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)  # Preprocess for VGG16
    return img_array

# Predict function
def predict_image(model, img_array, class_labels):
    predictions = model.predict(img_array)  # Predict
    predicted_class = np.argmax(predictions, axis=1)  # Get class with highest probability
    return class_labels[predicted_class[0]], predictions[0][predicted_class[0]]

if model_file is not None and image_file is not None:
    # Load the model
    model = load_model(model_file)

    # Save the image file and preprocess
    with open("uploaded_image.jpg", "wb") as f:
        f.write(image_file.getbuffer())
    
    img_array = preprocess_image("uploaded_image.jpg")
    
    # Define class labels
    class_labels = ['Class1', 'Class2', 'Class3']  # Replace with your actual labels

    # Predict and display result
    predicted_label, confidence = predict_image(model, img_array, class_labels)
    st.write(f"Predicted class: {predicted_label} with confidence {confidence:.2f}")

    # Display the image
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
else:
    st.warning("Please upload both a model and an image.")
