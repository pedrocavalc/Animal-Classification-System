# Animal Classification System with Explainable AI
Welcome to the Animal Classification System repository. This project aims to classify animals using a machine learning model and provides explanations for its predictions using the SHAP method. I've incorporated several modern tools and techniques from the MLOps domain to ensure seamless deployment, scalability, reproducibility, and monitoring of our model.

## Features
Animal Classification: Classify animals based on the given image.
Explainability with SHAP: For every prediction, get a detailed SHAP explanation to understand the decision-making process of the model.
MLOps Integration: Full-fledged MLOps practices with tools like MLflow, Jenkins for CI/CD, and Docker for containerization.
## Tech Stack
TensorFlow: The primary deep learning framework used to train our animal classification model.
Docker: To containerize our services, ensuring consistent environments and easy deployment.
FastAPI: A modern web framework to serve our model predictions.
Streamlit: An interactive tool for building web applications for machine learning and data science.
MLflow: For tracking experiments, packaging code into reproducible runs, and sharing and deploying models.
Jenkins: An open-source automation server, facilitating continuous integration and continuous delivery.
SHAP: A tool to make machine learning models more transparent and to explain their predictions.
### Quick Start
Setup: Ensure you have Docker installed.
bash
Copy code
docker-compose up

Access Streamlit UI: Open your browser and navigate to http://localhost:8501.

Making Predictions: Upload an animal image and get both the classification and SHAP explanation.

MLflow Tracking: Navigate to http://localhost:5000 to see tracked experiments.

## Directory Structure
bash
Copy code
.
├── app/                    # FastAPI application folder
│   ├── main.py
│   ├── model/
│   ├── components/
├── streamlit/              # Streamlit web application folder
├── Dockerfile
├── docker-compose.yml
├── Jenkinsfile             # Jenkins pipeline definition
└── README.md
# Development
## Model Training: I used TensorFlow to train our model. Training scripts are available in the app/model directory.

## API with FastAPI: The FastAPI application in the app/ directory serves the trained model. It accepts an image as input and returns the predicted class and SHAP explanation.

Frontend with Streamlit: The Streamlit application provides an interactive UI for users to upload images and view predictions and explanations.

## Continuous Integration and Continuous Deployment (CI/CD)
The Jenkins pipeline automates the process of testing and deploying our application. It ensures that any changes to the codebase don't break existing functionality and that our application is always deployed with the latest features and fixes.

# Contributing
Feel free to fork this repository, open issues, or submit PRs. For major changes, please open an issue first to discuss the change.
