from azureml.core import Workspace, Environment, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies

# Connect to the workspace
ws = Workspace.from_config(path="config.json")

# Get the registered model
model = Model(ws, 'naruto_classifier_v2')

# Create environment
env = Environment('naruto-env')

# Create a CondaDependencies object and add required packages
conda_dep = CondaDependencies()
conda_dep.add_pip_package("azureml-defaults")  # Azure ML dependencies
#conda_dep.add_conda_package("python=3.9")  # Example: Add specific Python version

# Read packages from requirements.txt and add them as pip packages
with open("requirements.txt", "r") as req_file:
    for line in req_file.readlines():
        conda_dep.add_pip_package(line.strip())  # Assuming each line contains a package specifier

# Assign the CondaDependencies object to the environment
env.python.conda_dependencies = conda_dep

print("All dependencies fixed")
# Inference config
inf_config = InferenceConfig(entry_script="score.py", environment=env)

# Deployment config
dep_config = AciWebservice.deploy_configuration(cpu_cores=2, memory_gb=4)

# Deploy the model
service = Model.deploy(workspace=ws, 
                       name="naruto-classifier-service-v2", 
                       models=[model], 
                       inference_config=inf_config, 
                       deployment_config=dep_config)

service.wait_for_deployment(show_output=True)

print(f"Service {service.name} deployed: {service.scoring_uri}")
