
#conda env create -f environment.yml

import json
from azureml.core import Workspace, Environment

class AzureMLEnvironmentManager:
    def __init__(self, config_path="config.json"):
        self.ws = Workspace.from_config(path=config_path)

    def register_environment_from_file(self, environment_name, file_path):
        # Create the environment from the conda specification
        env = Environment.from_conda_specification(name=environment_name, file_path=file_path)
        
        # Register the environment to the workspace
        env.register(workspace=self.ws)
        
        print(f"Environment {env.name} registered successfully!")
        return env

# Test section
if __name__ == "__main__":
    manager = AzureMLEnvironmentManager()
    environment_name = "my_custom_environment"
    file_path = "environment.yml"
    
    manager.register_environment_from_file(environment_name, file_path)
