import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

st.title("YOLO Polygon WebApp")
st.write("อัปโหลดรูปภาพ")


uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    st.image(image, caption='รูปต้นฉบับ', use_container_width=True)
    st.write("กำลังประมวลผล...")
    
    results = model.predict(source=img_array, conf=0.25)
    res_plotted = results[0].plot()
    
    final_img = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    st.image(final_img, caption='ผลลัพธ์จากโมเดล', use_container_width=True)