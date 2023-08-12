# MLOps: Registering and Deploying Endpoints in Azure

This README provides a step-by-step guide to registering and deploying machine learning endpoints in Azure. It also outlines the progress made so far and the main challenges faced.

## Progress

1. **Local and Azure Built-In VSCode Execution**
   - The model was successfully executed both locally and in the built-in VSCode environment provided by Azure.
   
2. **Model Registration and Prediction**
   - The model has been registered in Azure.
   - We were able to open the registered model and make predictions.

3. **Endpoint Production Issues**
   - Encountered problems while attempting to deploy to a production endpoint.
   - Encountered inconsistencies and missing YAML configuration files, leading to build failures.

> **Note**: Ensure that you update the configuration with your personal credentials before proceeding.

## Main Challenge: Docker Image Build Failure in Azure

The main challenge faced was the failure of the Docker image build process in Azure. The following steps were taken to address this issue:

1. **Environment Fixes**
   - Fixed existing environments and established a separate Conda environment for the deployment process.

2. **Environment Registration**
   - Created and registered a new environment in Azure, specifically for deployment purposes.

## Contribution

If you have any suggestions or fixes, feel free to contribute to this project. Your feedback is invaluable to improving the deployment process in Azure.

---

For any questions or further clarifications, please raise an issue or contact the project maintainers.
