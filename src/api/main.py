from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from components.components import load_model,  CLASS_DICT
from tensorflow.keras.applications.resnet50 import preprocess_input
from fastapi.responses import Response
import shap
import matplotlib.pyplot as plt
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


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


    image_array = np.expand_dims(image_array, axis=0)
    print(image_array.shape)  
    predictions = model.predict(image_array)
    masker = shap.maskers.Image(mask_value="blur(128, 128)", 
                            shape=(224,224,3))
    explainer = shap.Explainer(model=model.predict, 
                           masker=masker, 
                           algorithm="auto", 
                           output_names=list(CLASS_DICT.values()))
    shap_values = explainer(image_array, max_evals=1000, outputs=shap.Explanation.argsort.flip[:3])
    
    shap.image_plot(shap_values=shap_values, show=False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()


    return Response(content= buf.getvalue(), media_type="image/png")




