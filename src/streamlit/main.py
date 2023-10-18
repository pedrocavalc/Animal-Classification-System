import streamlit as st
import requests
from PIL import ImageDraw, Image
from streamlit.runtime.scriptrunner import add_script_run_ctx
from io import BytesIO
from annotated_text import annotated_text, annotation
import time
import threading
import base64

# Constants
API_URL = "http://localhost:8000/predict"
REFERENCE_RESOLUTION = 1920 * 1080
REFERENCE_SPEED = 0.01  
SQUARE_SIZE = 20
stop_scanning = threading.Event()

def simulate_scanning_effect(image: "PIL.Image.Image", placeholder):
    """
    Simulates scanning effect on the uploaded image.
    """
    width, height = image.size
    base = image.convert("RGBA")
    
    sleep_time = (width * height / REFERENCE_RESOLUTION) * REFERENCE_SPEED
    sleep_time = max(0.001, min(sleep_time, 0.1))
    
    is_horizontal_scan = True
    position = 0

    while not stop_scanning.is_set():
        overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        if is_horizontal_scan:
            for y_position in range(0, height, SQUARE_SIZE):
                draw.rectangle([position, y_position, position + SQUARE_SIZE, y_position + SQUARE_SIZE], fill=(70, 130, 255, 100))

            position += SQUARE_SIZE
            if position >= width:
                position = 0
                is_horizontal_scan = not is_horizontal_scan
        else:
            for x_position in range(0, width, SQUARE_SIZE):
                draw.rectangle([x_position, position, x_position + SQUARE_SIZE, position + SQUARE_SIZE], fill=(70, 130, 255, 100))

            position += SQUARE_SIZE
            if position >= height:
                position = 0
                is_horizontal_scan = not is_horizontal_scan

        combined_image = Image.alpha_composite(image.convert("RGBA"), overlay)
        placeholder.image(combined_image, caption='Scanning...', use_column_width=True)
        
        time.sleep(sleep_time)

def send_image_to_api(image) -> requests.Response:
    """
    Sends image to the API endpoint for prediction.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_str = buffered.getvalue()
    files = {"file": image_str}
    print(API_URL)
    return requests.post(API_URL, files=files)

def upload_function():
    """
    Manages the file uploading process and display predictions.
    """
    uploaded_file = st.file_uploader("Choose a file", type=['jpg','png'])
    
    if uploaded_file is None:
        st.warning("Please upload a file.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        placeholder = st.empty()
        
        stop_scanning.clear()
        scanning_thread = threading.Thread(target=simulate_scanning_effect, args=(image, placeholder))
        add_script_run_ctx(scanning_thread)
        scanning_thread.start()
        
        response = send_image_to_api(image)

        stop_scanning.set()
        scanning_thread.join()
        placeholder.empty()

        if response.status_code == 200:
            response_data = response.json()
            decoded_image = base64.b64decode(response_data["shap_image"].split(",")[1])
            prediction_image = Image.open(BytesIO(decoded_image))
            
            st.divider()               
            st.markdown("## Predicted class: ", unsafe_allow_html=True)
            annotated_text(
                annotation(
                    response_data['predicted_class'].capitalize(),
                    font_family="Arial",
                    border="1px solid gray",
                    font_size="24px",
                    font_weight="bold",
                    background="lightyellow",
                    color="black"
                )
            )
            st.image(prediction_image, use_column_width=True)


            st.markdown("## To make another prediction, upload another file above!")

        else:
            st.error('Error during API call')


def main():
    st.title("Welcome to Explainable AI app for animal classification!")
    st.divider()
    st.sidebar.title("Informations about the project:")
    st.sidebar.info("Developed by Pedro Cavalcante. [GitHub](https://github.com/pedrocavalc)")
    st.sidebar.info("Trained on the Animals-90 dataset using ResNet50. The project uses SHAP for predictions.")
    st.sidebar.info("Goal: Learn MLOps techniques for production. Models retrain every 3 days, with automated tests via Jenkins and deploy with MLFlow.")
    st.sidebar.info("For more details, check the [repository](https://github.com/pedrocavalc/Animal-Classification-System).")
    
    upload_function()


if __name__ == "__main__":
    main()
