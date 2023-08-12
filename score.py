import json
import os
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from azureml.core import Workspace
from azureml.core.model import Model as AzureModel

class NarutoClassifier:
    def __init__(self, config_path="config.json", model_name="naruto_classifier_v2", model_version=1):
        self.model_name = model_name
        self.model_version = model_version
        self.ws = Workspace.from_config(path=config_path)
        self._load_model()

    def _load_model(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Download the model
        model_obj = AzureModel(workspace=self.ws, name=self.model_name, version=self.model_version)
        model_path = model_obj.download(target_dir='.', exist_ok=True)

        # Load the model architecture
        base_model = ResNet50(weights=None, include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.6)(x)
        predictions = Dense(3, activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Load the saved weights
        self.model.load_weights(model_path)

    def predict(self, image_url):
        img_array = self._load_and_preprocess_image_from_url(image_url)
        class_probabilities = self.model.predict(img_array)
        predictions = np.argmax(class_probabilities, axis=1)

        class_names = ['Jiraiya', 'Sakura', 'Sasuke']
        return class_names[predictions[0]]

    def _load_and_preprocess_image_from_url(self, url):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))

        # Convert the image to RGB format
        image = image.convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)

# This init() function is required by the Azure ML Inference Server
def init():
    global model_instance
    model_instance = NarutoClassifier()

# This run() function is also required by the Azure ML Inference Server
def run(data):
    try:
        data = json.loads(data)
        result = model_instance.predict(data['url'])
        return json.dumps({"result": result})
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})

# Test section
if __name__ == "__main__":
    classifier = NarutoClassifier()

    test_image_url = 'https://www.pngitem.com/pimgs/m/653-6536166_sasuke-head-png-jpg-freeuse-sasuke-4th-great.png'
    result = classifier.predict(test_image_url)

    print(result)
