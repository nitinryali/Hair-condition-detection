import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('model.pkl')

# Define the labels for the classes
labels = ['damage', 'high damage', 'weak damage']

def preprocess_image(image):
    image = image.resize((180, 180))
    image_array = np.array(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0 - 0.5) * 2.0
    return np.expand_dims(normalized_image_array, axis=0)

def app():
    st.title("Hair condition detection")
    uploaded_file = st.file_uploader('Upload a SEM(Microscopic) image', type=['jpg', 'jpeg', 'png'])
    st.write("Check out this [link](https://en.wikipedia.org/wiki/Scanning_electron_microscope) for input image and to know about SEM Images")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Detect Hair Condition'):
            preprocessed_image = preprocess_image(image)
            predictions = model.predict(preprocessed_image)
            predicted_label = labels[np.argmax(predictions)]

            st.subheader("Your Hair Condition is " + predicted_label)

# Run the app
if __name__ == '__main__':
    app()
