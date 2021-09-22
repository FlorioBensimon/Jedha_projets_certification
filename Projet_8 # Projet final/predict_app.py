import tensorflow as tf
import base64
import numpy as np 
from PIL import Image
import io
from tensorflow import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS
from flask import Flask, render_template


app = Flask(__name__)
CORS(app)
 
def get_model():
    global model
    model = load_model('rappers.h5')
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.expand_dims(image, 0)
    image = image/255.
    
    return image

print(" * Loading Keras Model...")
get_model()

@app.route('/predict', methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message ['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    
    class_names = ['21 Savage', 'ASAP Rocky', 'Action Bronson', 'Azaelia Banks', 'Bad Bunny', 'Beyonce', 'Cardi b', 'Chance The Rapper', 'Da Baby', 'Doja cat', 'Drake', 'Eminem', 'Future', 'G-eazy', 'Juice WRLD', 'Justin Bieber', 'Kanye West', 'Katy Perry', 'Kodak Black', 'Lil Mosey', 'Lil Peep', 'Lil baby', 'Mac Miller', 'Machine Gun Kelly', 'Macklemore', 'Madonna', 'Nicki Minaj', 'Playboi Carti', 'Post Malone', 'Rihanna', 'Russ', 'SixNine', 'Taylor Swift', 'Travis Scott', 'Trippie Redd', 'XXX Tentacion']
    
    pred = model.predict(processed_image).tolist()
    prediction = class_names[np.argmax(pred)]
    
    response = {
        "prediction" :  prediction
    }
    print(prediction)
    return jsonify(response)
