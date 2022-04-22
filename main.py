import uvicorn
from fastapi import FastAPI,File,UploadFile
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

app = FastAPI()

MODEL = tf.keras.models.load_model('models\elon_musk_amar_rahe.h5')
CLASS_NAMES = ['neutral','happy','sad']

@app.get('/ping')
async def index():
    return {'message':'Hello, Harsh'}

def read_file_as_image(data) -> np.ndarray:
    img = np.array(Image.open(BytesIO(data)))
    return img

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.1,4)
    face_roi = None
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey:(ey+eh), ex:(ex+eh)]
    
    img_size = 224
    final_image = cv2.resize(face_roi,(img_size,img_size))
    final_image = np.expand_dims(final_image,axis= 0)
    final_image = final_image/255.0

    Predictions = MODEL.predict(final_image)
    p = CLASS_NAMES[np.argmax(Predictions[0])]

    return {"Emotion" : p } 


if __name__ == '__main__':
    uvicorn(app, host = '127.0.0.1',port=8000)