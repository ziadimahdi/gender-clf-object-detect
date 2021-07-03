# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:14:11 2020

@author: abc
"""
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from apps.img_classification import teachable_machine_classification
from PIL import Image
import awesome_streamlit as ast



def write():
    """Method used to write the page in the app.py file"""
    ast.shared.components.title_awesome("Gender Classification & Object Detection")
    with st.spinner("Loading  ..."):



        st.header("Gender Classification")

        image = Image.open('apps/gender-equality.jpg')
        st.image(image, caption='Gender Classification',width = 400)

        st.text("Upload a Picture for image classification as MALE or FEMALE")
        uploaded_file = st.file_uploader("Choose a Picture ...", type="jpg")



        if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Picture.',width = 200)
                st.write("")
                st.write("Classifying...")

                label = teachable_machine_classification(image, 'apps/keras_model.h5')
                if label == 0:
                    st.write("The IMAGE scan is for MALE")

                else:
                    st.write("The IMAGE scan is for FEMALE")
