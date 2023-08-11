import json
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from azureml.core import Workspace
from azureml.core.model import Model as AzureModel
import os
def init():

    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    global model

    # Connect to the workspace
    ws = Workspace.from_config(path="config.json")  # Ensure you have the config.json in your directory or provide the full path.

    # Download the model
    model_obj = AzureModel(workspace=ws, name="naruto_classifier_v2", version=1)
    model_path = model_obj.download(target_dir='.', exist_ok=True)

    # Load the model architecture
    base_model = ResNet50(weights=None, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.6)(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Load the saved weights
    model.load_weights(model_path)


def run(raw_data):
    data = json.loads(raw_data)['data']
    img_array = load_and_preprocess_image_from_url(data)
    class_probabilities = model.predict(img_array)
    predictions = np.argmax(class_probabilities, axis=1)
    
    class_names = ['Jiraiya', 'Sakura', 'Sasuke']
    return class_names[predictions[0]]

def load_and_preprocess_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    
    # Convert the image to RGB format
    image = image.convert('RGB')
    
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
    result = run(json.dumps({"data": 'https://www.pngitem.com/pimgs/m/653-6536166_sasuke-head-png-jpg-freeuse-sasuke-4th-great.png'}))
    print(result)
