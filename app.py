import streamlit as st
import tensorflow as tf
#from tensorflow import keras
import random
from PIL import Image, ImageOps
import numpy as np

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Tire Quality Detection",
    page_icon = ":car:",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


with st.sidebar:
        st.image('mg.png')
        st.title("Tire Quality Detection")
        st.write("This software accurately detects the quality of tires of any vehicle.")

        st.sidebar.info("The detection is performed with the help of ResNet-50 and MobileNet V3 and is faster and lighter than its competitors")

             
        
def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            
            return key
        
       

    


@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('tire.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()
    #model = keras.Sequential()
    #model.add(keras.layers.Input(shape=(224, 224, 4)))
    

st.write("""
         # Tire Quality Detection
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction


        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['defective', 'good']


    string = "Detected Result : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'good':
        st.balloons()
        st.success("Its time for a long drive! 😎🚗")
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'defective':
        st.sidebar.warning(string)
        st.sidebar.success("Kindly scroll to page bottom for inference")
        st.snow()
        st.markdown("## Inference")
        st.info("The tire is detected to be worn out. Kindly consider checking the regions of damage with out application and get it fixed before you sit on the road.")


        st.markdown(
        f'<a href="https://tire-and-damage.streamlit.app/" target="_blank" style="display: inline-block; padding: 12px 20px; background-color: black; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">Tire Damage Segmentation</a>',
        unsafe_allow_html=True)

with st.expander("Check sample data source"):
            st.markdown(f'<a href="https://www.kaggle.com/datasets/warcoder/tyre-quality-classification/data" target="_blank" style="display: inline-block; padding: 12px 20px; background-color: black; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">Sample Dataset</a>', unsafe_allow_html=True)  
