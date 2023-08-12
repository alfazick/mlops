# wget https://github.com/alfazick/narutoclassifier/raw/main/models/model_weights_only.h5

import json
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

# Load the configuration from config.json
with open('config.json', 'r') as f:
    config = json.load(f)

subscription_id = config["subscription_id"]
resource_group_name = config["resource_group"]
workspace_name = config["workspace_name"]

# Connect to Azure ML
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group_name, workspace_name
)

# Ensure we have the correct path to the model file
model_path = "./model_weights_only.h5"

# Use the Model entity to structure our registration
model_to_register = Model(
    name="naruto_classifier_v2",
    description="Naruto Character Classifier",
    path=model_path,
    version=None,  # It will auto-increment if set to None
)

# Register the model using the create_or_update method
registered_model = ml_client.models.create_or_update(model_to_register)

print(f"Model {registered_model.name} version {registered_model.version} registered.")
