import json
import numpy as np
from tensorflow.keras.models import load_model
from azureml.core.model import Model
from PIL import Image
import requests
from io import BytesIO

def init():
    global model
    #model_path = Model.get_model_path('naruto_classifier_v2')
    model_path = Model.get_model_path(model_name="naruto_classifier_v2", version=3)

    model = load_model(model_path)

def run(raw_data):
    data = json.loads(raw_data)['data']
    img_array = load_and_preprocess_image_from_url(data)
    class_probabilities = model.predict(img_array)
    predictions = np.argmax(class_probabilities, axis=1)
    return predictions[0]

def load_and_preprocess_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


# Test section
if __name__ == "__main__":
    init()  # Initialize the model

    # Download and read the image
    response = requests.get('https://www.pngitem.com/pimgs/m/653-6536166_sasuke-head-png-jpg-freeuse-sasuke-4th-great.png')
    img = Image.open(BytesIO(response.content))

    # Process the image
    img = img.resize((224, 224))  # Adjust the size based on your model's input size
    img_array = np.array(img)

    # Use the processed image as input and get the prediction
    result = run(img_array)
    print(result)