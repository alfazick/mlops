from azureml.core import Workspace, Environment, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

# Connect to the workspace
ws = Workspace.from_config(path="config.json")

# Get the registered model
model = Model(ws, 'naruto_classifier_v2')

# Create environment
env = Environment('naruto-env')
python_packages = ['numpy', 'tensorflow', 'jsonpickle==1.5.2']
for package in python_packages:
    env.python.conda_dependencies.add_pip_package(package)

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

print(f"Service {service_name} deployed: {service.scoring_uri}")
