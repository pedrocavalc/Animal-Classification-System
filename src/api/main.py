from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from PIL import Image
import sys
sys.path.append('src/')

app = FastAPI()

model = tf.keras.models.load_model('model/model.h5')

@app.get("/")
async def root():
    return 'Hello World!'

@app.post("/predict")
async def predict(file:UploadFile):
    pass