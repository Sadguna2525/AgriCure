import streamlit as st
import numpy as np
from PIL import Image
import cv2 as cv
import keras


label_name = {
    0: 'Apple_Apple_scab',
    1: 'Apple_Black_rot',
    2: 'Apple_Cedar_apple_rust',
    3: 'Apple_healthy',
    4: 'Blueberry_healthy',
    5: 'Cherry_including_sour_Powdery_mildew',
    6: 'Cherry_including_sour_healthy',
    7: 'Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot',
    8: 'Corn_maize_Common_rust',
    9: 'Corn_maize_Northern_Leaf_Blight',
    10: 'Corn_maize_healthy',
    11: 'Grape_Black_rot',
    12: 'Grape_Esca_Black_Measles',
    13: 'Grape_Leaf_blight_Isariopsis_Leaf_Spot',
    14: 'Grape_healthy',
    15: 'Orange_Huanglongbing_Citrus_greening',
    16: 'Peach_Bacterial_spot',
    17: 'Peach_healthy',
    18: 'Pepper_bell_Bacterial_spot',
    19: 'Pepper_bell_healthy',
    20: 'Potato_Early_blight',
    21: 'Potato_Late_blight',
    22: 'Potato_healthy',
    23: 'Raspberry_healthy',
    24: 'Soybean_healthy',
    25: 'Squash_Powdery_mildew',
    26: 'Strawberry_Leaf_scorch',
    27: 'Strawberry_healthy',
    28: 'Tomato_Bacterial_spot',
    29: 'Tomato_Early_blight',
    30: 'Tomato_Late_blight',
    31: 'Tomato_Leaf_Mold',
    32: 'Tomato_Septoria_leaf_spot',
    33: 'Tomato_Spider_mites_Two_spotted_spider_mite',
    34: 'Tomato_Target_Spot',
    35: 'Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato_Tomato_mosaic_virus',
    37: 'Tomato_healthy'
}

d = {
    'Apple_Apple_scab': 'Captan, mancozeb, sulfur',
    'Apple_Black_rot': 'Thiophanate-methyl, myclobutanil',
    'Apple_Cedar_apple_rust': 'Myclobutanil, sulfur-based',
    'Cherry_including_sour_Powdery_mildew': 'Sulfur-based fungicides',
    'Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot': 'Azoxystrobin, propiconazole',
    'Corn_maize_Common_rust': 'Mancozeb, chlorothalonil',
    'Corn_maize_Northern_Leaf_Blight': 'Pyraclostrobin, trifloxystrobin',
    'Grape_Black_rot': 'Sulfur, copper, mancozeb',
    'Grape_Esca_Black_Measles': 'Tebuconazole (limited effect)',
    'Grape_Leaf_blight_Isariopsis_Leaf_Spot': 'Copper-based fungicides',
    'Orange_Huanglongbing_Citrus_greening': 'Imidacloprid (vector control)',
    'Peach_Bacterial_spot': 'Copper-based bactericides',
    'Pepper_bell_Bacterial_spot': 'Copper-based sprays',
    'Potato_Early_blight': 'Mancozeb, chlorothalonil',
    'Potato_Late_blight': 'Metalaxyl, cymoxanil',
    'Squash_Powdery_mildew': 'Sulfur, potassium bicarbonate',
    'Strawberry_Leaf_scorch': 'Captan, myclobutanil',
    'Tomato_Bacterial_spot': 'Copper-based bactericides',
    'Tomato_Early_blight': 'Chlorothalonil, azoxystrobin',
    'Tomato_Late_blight': 'Metalaxyl, mancozeb',
    'Tomato_Leaf_Mold': 'Chlorothalonil, copper hydroxide',
    'Tomato_Septoria_leaf_spot': 'Chlorothalonil, mancozeb',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Abamectin, insecticidal soap',
    'Tomato_Target_Spot': 'Azoxystrobin, chlorothalonil',
    'Tomato_Yellow_Leaf_Curl_Virus': 'Imidacloprid (vector)',
    'Tomato_Tomato_mosaic_virus': 'It is virus'
}

d1 = {
    'Apple_Apple_scab': 'Prune trees, remove fallen leaves',
    'Apple_Black_rot': 'Remove infected fruit, pruning',
    'Apple_Cedar_apple_rust': 'Remove nearby junipers, use resistant cultivars',
    'Cherry_including_sour_Powdery_mildew': 'Remove infected leaves, prune trees',
    'Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot': 'Crop rotation, residue management',
    'Corn_maize_Common_rust': 'Resistant varieties, rotation',
    'Corn_maize_Northern_Leaf_Blight': 'Remove debris, use resistant hybrids',
    'Grape_Black_rot': 'Remove infected canes, prune canopy',
    'Grape_Esca_Black_Measles': 'Remove infected vines, avoid pruning wounds',
    'Grape_Leaf_blight_Isariopsis_Leaf_Spot': 'Good air circulation, leaf removal',
    'Orange_Huanglongbing_Citrus_greening': 'Remove infected trees, biological psyllid control',
    'Peach_Bacterial_spot': 'Biocontrol with nonpathogenic Xanthomonas',
    'Pepper_bell_Bacterial_spot': 'Disease-free seeds, no overhead irrigation',
    'Potato_Early_blight': 'Neem extract, Trichoderma viride',
    'Potato_Late_blight': 'Certified seeds, Trichoderma application',
    'Squash_Powdery_mildew': 'Resistant varieties, good airflow',
    'Strawberry_Leaf_scorch': 'Remove infected leaves, space plants',
    'Tomato_Bacterial_spot': 'Disease-free seeds, rotation',
    'Tomato_Early_blight': 'Clonostachys spp. biocontrol',
    'Tomato_Late_blight': 'Crop rotation, certified seeds',
    'Tomato_Leaf_Mold': 'Ventilation, resistant varieties',
    'Tomato_Septoria_leaf_spot': 'Crop debris removal, staking',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Predatory mites, neem oil',
    'Tomato_Target_Spot': 'Crop debris management, avoid overcrowding',
    'Tomato_Yellow_Leaf_Curl_Virus': 'Remove infected plants, control whiteflies',
    'Tomato_Tomato_mosaic_virus': 'Sanitize tools, resistant varieties'
}
      
model = keras.models.load_model("C:\\Users\\MEGHANA\\Downloads\\AgriCure.h5")


def load_preprocess_image(uploaded_image, target_size=(224, 224)):
    img = Image.open(uploaded_image)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0
    return img_array

def predict_disease(model, uploaded_image, class_indices):
    preprocess_img = load_preprocess_image(uploaded_image)
    prediction = model.predict(preprocess_img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class = class_indices[predicted_class_index]
    return predicted_class

st.title("AGRICURE")


uploaded_image = st.file_uploader("Upload a leaf image",type=["jpg", "jpeg", "png"])

    
if uploaded_image is not None:
    st.image(uploaded_image)
    result = predict_disease(model, uploaded_image, label_name)
    st.write("Predicted Disease:", result)
    
    if result in d:
        st.write(f"Recommended pesticide is:{d[result]}")
    
    
    if result in d1:
        st.write("Alternate Biological practices:", d1[result])

