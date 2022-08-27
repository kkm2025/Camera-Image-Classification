import streamlit as st

#def add_bg_from_url():
#    st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("https://c4.wallpaperflare.com/wallpaper/410/867/750/vector-forest-sunset-forest-sunset-forest-wallpaper-preview.jpg");
#             background-attachment: fixed;
#             background-size: cover
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#add_bg_from_url() 
#https://media.istockphoto.com/illustrations/light-blue-abstract-background-of-stone-wall-illustration-id665098636?k=20&m=665098636&s=612x612&w=0&h=er9yIl5lH2v1dhUpl19cGxsuuZwxx4aF5HSCU1YxTD8=
#https://images.freecreatives.com/wp-content/uploads/2016/04/Elegant-Solid-Yellow-Backgrounds-.jpg

from PIL import Image ,ImageOps
#import PIL
import tensorflow as tf 
import tensorflow_hub as hub 
import urllib.request
import numpy as np
import time
st.set_page_config(
     page_title="Camera Classification",
     page_icon="📷",
     layout="wide",
 )
st.markdown("""# Image Classification""")

tab1,tab2= st.tabs(["About","Prediction"])
with tab1:
    st.image("image_webap.webp")
    st.write("Image classification is the process of categorizing and labeling groups of pixels or vectors within an image based on specific rules. The categorization law can be devised using one or more spectral or textural characteristics. ")
    st.subheader("What is transfer learning?")
    st.write("Transfer learning is a research problem in the field of machine learning. It stores the knowledge gained while solving one problem and applies it to a different but related problem. For example, the knowledge gained while learning to recognize cats could apply when trying to recognize cheetahs. In deep learning, transfer learning is a technique whereby a neural network model is first trained on a problem similar to the problem that is being solved.")
    st.text("For this project Mobilenet_v2_050_224 pre-trained model was used.")
    st.subheader("MobilenetV2")
    st.write("""
             MobileNetV2 is a classification model developed by Google. It provides real-time classification capabilities under computing constraints in devices like smartphones. 
             This implementation leverages transfer learning from ImageNet to your dataset.
             """)
    st.write("Know more about Mobilenet V2 at [link](%s)"%"https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/5")
    url = "https://github.com/kkm2025/Camera-Image-Classification"
    st.write("[Project Github Repository](%s)" % url)
with tab2:
    st.subheader('Classifying Images containing mobile phones and digital camera')
    #https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/5
    model = tf.keras.models.load_model("mobilenet_last_good_acc.h5",custom_objects={"KerasLayer":hub.KerasLayer})

    Labels = ["Digital Camera", "Phone"]
    st.info("Please upload those images which contains phone or digital camera.", icon="ℹ️")
    uploaded_file = st.file_uploader("Choose a image ",type=["jpg",'png','jpeg','webp'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = image
        #image sizing
        size = (224, 224)
        try:
          image = ImageOps.fit(image, size, Image.ANTIALIAS)
        except ValueError as ve:
          st.warning('The selected image cannot be used for classification due to size issues.Please select any other image', icon="⚠️")

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
            with st.spinner('Classifying.....'):
                time.sleep(4)
                #st.success('Done!')
            with st.spinner("Almost there..."):
                time.sleep(2)
            #my_bar = st.progress(0)
            #for percent_complete in range(100):
            #    time.sleep(0.1)
            #    my_bar.progress(percent_complete + 1)
            make_prediction = model.predict(data)
            label = Labels[np.argmax(make_prediction)]
            st.success("The image contains "+ str.lower(label) + " with probability " + str(np.max(make_prediction)*100)[:5]+"%")
            st.snow()
    
    st.caption("Created by Kirankumar Manda")   

    

