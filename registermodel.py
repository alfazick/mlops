# wget https://github.com/alfazick/narutoclassifier/raw/main/models/model_weights_only.h5

from azureml.core import Workspace, Model

# Connect to the workspace
ws = Workspace.from_config()

# Register the model
model = Model.register(workspace=ws,
                       model_name='naruto_classifier_v2',  # Provide a different model name for clarity
                       model_path='./model_weights_only.h5',  # Path to the TensorFlow model weights
                       description='Naruto Character Classifier')

print(f"Model {model.name} version {model.version} registered.")
