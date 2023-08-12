import os
import numpy as np
import tensorflow.keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
import requests
from PIL import Image
from io import BytesIO

# Define the classes (adjust based on your model's classes)
classes = ['Jiraiya', 'Sakura', 'Sasuke']  # Replace with your class names if different

# Recreate the model architecture
base_model = ResNet50(weights=None, include_top=False)  # Set weights to None

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)  # Removed regularization for simplicity
x = Dropout(0.6)(x)
predictions = Dense(len(classes), activation='softmax')(x)  # Adjusted based on number of classes
model = Model(inputs=base_model.input, outputs=predictions)

# Load the saved weights
weights_file_name = 'model_weights_only.h5'
model.load_weights(weights_file_name)

# Print the summary of the model (optional)
#model.summary()

# Fetch the image
#response = requests.get('https://www.pngitem.com/pimgs/m/653-6536166_sasuke-head-png-jpg-freeuse-sasuke-4th-great.png')
response = requests.get('https://raw.githubusercontent.com/alfazick/mlops/main/HD-wallpaper-jiraiya-naruto-jiraiya.jpg')
img = Image.open(BytesIO(response.content))

# Convert to RGB format
img = img.convert('RGB')

# Resize the image
img = img.resize((224, 224))

# Convert to numpy array
img_array = np.array(img)

# Optionally normalize (assuming values between 0 and 1)
img_array = img_array / 255.0

# Expand dimensions for model prediction (since the model expects a batch)
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)

# Get class with highest probability
predicted_class = np.argmax(predictions[0])

# Map to actual class names (assuming you have 3 classes as previously mentioned)
class_names = ['Jiraiya', 'Sakura', 'Sasuke']
print(f"The predicted class is: {class_names[predicted_class]} with a confidence of {predictions[0][predicted_class]*100:.2f}%")
