import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
model_path=os.getenv("model_path1")

# Define the categories for classification
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Define image dimensions
img_height = 180
img_width = 180

# Define the path to your model
#model_path = 'D:\Deep learning\Image_classification-Deep-Learning-\Fruits_Vegetables\Image_classify.keras'  # Update with the actual path

# Initialize model variable
model = None

# Check if model file exists and load the model
if os.path.isfile(model_path):
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error(f"Model file not found at path: {model_path}")

# Streamlit app
st.header('Image Classification Model')

# Input image path
image_path = st.text_input('Enter Image Name', 'Apple.jpg')

if model is not None:
    # Load and preprocess the image
    try:
        image = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.img_to_array(image)  # Convert image to array
        img_bat = np.expand_dims(img_arr, axis=0)    # Create a batch

        # Predict and calculate scores
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict[0])

        # Display results
        st.image(image, caption='Uploaded Image')
        st.write('Vegetable/Fruit in image is: {}'.format(data_cat[np.argmax(score)]))
        st.write('With an accuracy of: {:.2f}%'.format(np.max(score) * 100))

    except Exception as e:
        st.write("Error processing image: ", e)
