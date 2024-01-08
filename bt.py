
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from tensorflow.keras.utils import load_img, img_to_array
#from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model

model = load_model('VGG_model.h5')

classes = {0:'glioma',1:'meningioma', 2:'pituitary'}
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Brain Tumor Classification", page_icon=":brain:", layout="wide", initial_sidebar_state="collapsed")

# Sidebar navigation
nav_selection = st.sidebar.radio("Navigation", ["Main", "About"])

# Main page
if nav_selection == "Main":
    st.title("Brain Tumor Classification")
    

    img_file = st.file_uploader('Select an image', type=['jpg','png','jpeg','gif','jfif','heic'])

    if img_file is not None :
        img = Image.open(img_file)
        img = img.convert("RGB")  # Convert image to RGB format
        st.image(img,caption='Upload image succesfully')

    if st.button('predict'):
        img = img.resize((256,256))
        i = img_to_array(img)
        i = preprocess_input(i)
        input_arr = np.array([i])
        
        y_out = np.argmax(model.predict(input_arr))
        y_out1 = classes[y_out]
        
        #if y_out1 = 0:
        st.write(f'This image is a {y_out1}')

# About page
elif nav_selection == "About":
    st.title("About Brain Tumor Classification")
    
    img = Image.open('image.jfif')
    img = img.resize((100,40))
    st.image(img, caption='Example image of a brain', use_column_width=True)

    st.write("Brain tumor classification is a crucial task in medical diagnosis and treatment planning. It involves categorizing brain tumors based on their characteristics, which helps in determining the appropriate treatment approach. There are several types of brain tumors, including gliomas, meningiomas, pituitary adenomas, and medulloblastomas.")
    
    st.write("Gliomas are the most common type of brain tumor and originate from the glial cells in the brain. They can be further classified into different grades based on their aggressiveness, with grade IV glioblastoma being the most malignant. Meningiomas, on the other hand, develop from the meninges, which are the protective membranes surrounding the brain and spinal cord. Pituitary adenomas are tumors that develop in the pituitary gland, which is located at the base of the brain and plays a key role in regulating various bodily functions.")
