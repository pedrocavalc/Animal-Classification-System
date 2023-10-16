import streamlit as st
import requests
from PIL import ImageDraw, Image
from streamlit.runtime.scriptrunner import add_script_run_ctx
from io import BytesIO
from annotated_text import annotated_text, annotation
import time
import threading
import base64

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
        placeholder.empty()

        if response.status_code == 200:
            response_data = response.json()  # Parse JSON response
            decoded_image = base64.b64decode(response_data["shap_image"].split(",")[1])
            prediction_image = Image.open(BytesIO(decoded_image))
            annotated_text(f"Predicted class: ", annotation(response_data['predicted_class'], font_family="Comic Sans MS", border="1px dashed red"))
            st.image(prediction_image, caption=f"Prediction: {response_data['predicted_class']}", use_column_width=True)
            
        else:
            st.error('Error during API call')

def send_image_to_api(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_str = buffered.getvalue()
    files = {"file": image_str}
    response = requests.post(API_URL, files=files)
    return response

def main():
    st.title("Welcome to Explainable AI app for animal classification!")
    st.sidebar.title("Informations about the project:")
    st.sidebar.info("This project was developed by Pedro Cavalcante. [GitHub](https://github.com/pedrocavalc)")
    st.sidebar.info("The project uses a ResNet50 model trained on the Animals-90 dataset.The objective of the project is to classify images of animals into 90 differentes species. The project uses SHAP to explain the model's predictions.")
    st.sidebar.info("The main objective of the project is to learn MLOps techniques to serve the model in production time. Every 3 days a new model is automatically trained, the mlflow library is used to serve the models and automated tests with Jenkins to maintain the quality of the code and model before serving it to production.     If the trained model meets the necessary requirements, it is automatically sent to production and the old one archived.")
    st.sidebar.info("For more information about the tools used and the project flowchart, see the repository.")
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

    if st.button("Show Upload Widget", key="upload"):
        st.session_state.button_clicked = True

    if st.session_state.button_clicked:
        upload_function()

if __name__ == "__main__":
    main()