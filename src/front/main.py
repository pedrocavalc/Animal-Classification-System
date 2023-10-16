import streamlit as st
import requests
from PIL import ImageDraw, Image
from streamlit.runtime.scriptrunner import add_script_run_ctx
from io import BytesIO
import random
import time
import threading

API_URL = "http://localhost:8000/predict"
stop_scanning = threading.Event()

import streamlit as st
import requests
from PIL import ImageDraw, Image, ImageChops
from streamlit.runtime.scriptrunner import add_script_run_ctx
from io import BytesIO
import random
import time
import threading

API_URL = "http://localhost:8000/predict"
stop_scanning = threading.Event()

def simulate_scanning_effect(image: "PIL.Image.Image", placeholder):
    width, height = image.size
    square_size = 20

    base = image.convert("RGBA")
    
    # Parâmetros de velocidade
    reference_resolution = 1920 * 1080
    current_resolution = width * height
    reference_speed = 0.01  
    sleep_time = (current_resolution / reference_resolution) * reference_speed
    sleep_time = max(0.001, min(sleep_time, 0.1))  

    is_horizontal_scan = True
    position = 0

    while not stop_scanning.is_set():
        overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        if is_horizontal_scan:
            for y_position in range(0, height, square_size):
                draw.rectangle([position, y_position, position + square_size, y_position + square_size], fill=(70, 130, 255, 100))

            position += square_size
            if position >= width:
                position = 0
                is_horizontal_scan = not is_horizontal_scan

        else:
            for x_position in range(0, width, square_size):
                draw.rectangle([x_position, position, x_position + square_size, position + square_size], fill=(70, 130, 255, 100))

            position += square_size
            if position >= height:
                position = 0
                is_horizontal_scan = not is_horizontal_scan

        combined_image = Image.alpha_composite(image.convert("RGBA"), overlay)
        placeholder.image(combined_image, caption='Scanning...', use_column_width=True)
        
        time.sleep(sleep_time)

def upload_function():    
    uploaded_file = st.file_uploader("Choose a file", type=['jpg'])
    if uploaded_file is None:
        st.write("Please upload a file.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        placeholder = st.empty()
        
        stop_scanning.clear()
        thread = threading.Thread(target=simulate_scanning_effect, args=(image, placeholder))
        add_script_run_ctx(thread)
        thread.start()
        
        response = send_image_to_api(image)

        stop_scanning.set()
        thread.join()
        
        if response.status_code == 200:
            prediction_image = BytesIO(response.content)
            st.image(prediction_image, caption="Prediction", use_column_width=True)
        else:
            st.error("Error during API call.")

def send_image_to_api(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_str = buffered.getvalue()
    files = {"file": image_str}
    response = requests.post(API_URL, files=files)
    return response

def main():
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

    if st.button("Show Upload Widget", key="upload"):
        st.session_state.button_clicked = True

    if st.session_state.button_clicked:
        upload_function()

if __name__ == "__main__":
    main()