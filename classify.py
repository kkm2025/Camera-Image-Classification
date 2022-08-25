import streamlit as st

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://c4.wallpaperflare.com/wallpaper/410/867/750/vector-forest-sunset-forest-sunset-forest-wallpaper-preview.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 
#https://media.istockphoto.com/illustrations/light-blue-abstract-background-of-stone-wall-illustration-id665098636?k=20&m=665098636&s=612x612&w=0&h=er9yIl5lH2v1dhUpl19cGxsuuZwxx4aF5HSCU1YxTD8=
#https://images.freecreatives.com/wp-content/uploads/2016/04/Elegant-Solid-Yellow-Backgrounds-.jpg

from PIL import Image ,ImageOps
#import PIL
import tensorflow as tf 
import tensorflow_hub as hub 
import urllib.request
import numpy as np



st.title("Image Classification")
st.subheader('Classifying Images containing mobile phones and digital camera')
st.text("For this project mobilenet_v2_050_224 pre-trained model was used")#https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/5

model = tf.keras.models.load_model("mobilenet_last_good_acc.h5",custom_objects={"KerasLayer":hub.KerasLayer})

Labels = ["Digital Camera", "Phone"]

uploaded_file = st.file_uploader("Choose a image ",type=["jpg",'png','jpeg','webp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = image
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array
    #test_img = np.array(image).reshape(-1,224,224,3)
    new_image = image.resize((600, 400))
    st.image(new_image,caption="Uploaded image")
    #st.image(image,caption="Uploaded image",use_column_width='auto')
    if st.button("Predict the Class "):
        st.write('')
        st.write("Classifying....")
        make_prediction = model.predict(data)
        label = Labels[np.argmax(make_prediction)]
        st.success("The image contains "+ str.lower(label) + " with probability " + str(np.max(make_prediction)*100)[:5]+"%")

        
st.caption("Created by Kirankumar Manda")
