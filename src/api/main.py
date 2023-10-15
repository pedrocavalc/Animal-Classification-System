from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from components.components import load_model,  CLASS_DICT
from tensorflow.keras.applications.resnet50 import preprocess_input


from PIL import Image
import numpy as np
import io
import sys
sys.path.append('src/')

app = FastAPI()

model = load_model('model/model.h5')

@app.post("/predict")
async def predict(file:UploadFile):
    image_data = await file.read()
    
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    image = image.resize((224, 224)) 
    image_array = np.asarray(image)

    image_array = preprocess_input(image_array)

    image_array = np.expand_dims(image_array, axis=0)  
    predictions = model.predict(image_array)
    
    predicted_class = np.argmax(predictions, axis=1)

    predicted_class = CLASS_DICT[int(predicted_class[0])]
    
    return {"prediction": predicted_class}





