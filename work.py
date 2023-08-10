from azureml.core import Workspace
from azureml.core.webservice import Webservice

ws = Workspace.from_config()

# List all the deployed services
print("List of services in the workspace:")
for svc in Webservice.list(ws):
    print(svc.name)


service = Webservice(name="naruto-classifier-service-v2", workspace=ws)
print(service.state)

print(service.get_logs())
