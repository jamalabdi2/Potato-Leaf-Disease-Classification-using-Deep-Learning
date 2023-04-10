from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
from tensorflow import keras


app = FastAPI()
model_path = '/Users/jamal/Desktop/Potato leaf disease/saved_model/potato.h5'
def loadmodel(path:str):
    if not os.path.isfile(path):
        raise Exception(f'Model file {path} not found')
    try:
        model = tf.keras.models.load_model(path)
    except OSError as e:
        raise Exception (f'Erro loading model from {path}: {e}')

MODEL = loadmodel(model_path)
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


@app.get('/ping')
async def ping():
    return 'Hello, I am alive'


def byte_to_array(byte) -> np.ndarray:
    read_byte = BytesIO(byte)
    pillow_image = Image.open(read_byte)
    img = np.array(pillow_image)
    return img


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return {'error': 'File type not supported.'}
    image = byte_to_array(await file.read())
    img_batch = np.expand_dims(image, axis=0)
    prediction = MODEL.predict(img_batch)
    class_index = np.argmax(prediction[0])
    class_name = CLASS_NAMES[class_index]
    return {'class': class_name}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
