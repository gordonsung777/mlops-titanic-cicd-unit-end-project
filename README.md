This is my Solution to task 4 (look at number 4 to read the problem)


Predictive Modeling and Deployment
Problem Scenario

In this project, you will implement a complete MLOps lifecycle by building and deploying a machine learning model to classify patient readmission risk. This includes data exploration, model building, versioning, CI/CD pipeline development, REST API deployment, and monitoring using Python-based tools. This hands-on experience will strengthen your ability to deliver scalable, production-ready ML systems in real-world healthcare scenarios.

 

Instructions:  

Tools: Python, FastAPI, MLflow, DVC, Evidently AI
Dataset: Titanic dataset containing demographic and travel data
Submission: Submit all Python scripts, API code, monitoring reports, and a project report through the LMS

Situation:

You are an MLOps engineer working at a healthcare-focused analytics firm. Your team is tasked with building and maintaining an end-to-end ML pipeline to predict the risk of hospital readmission within 30 days. The project requires creating a fully automated, monitored, and version-controlled system that supports continuous integration and deployment. 


Task
Use Python and associated tools to develop an MLOps lifecycle project that includes the following:

Data exploration and preprocessing
Classification model development
Model and data versioning
CI/CD pipeline creation
Deployment of the model via FastAPI
Continuous monitoring using Evidently AI
Documentation of the entire workflow and best practices

Input dataset: DatasetLinks to an external site.

Actions 

Data Exploration and Preprocessing
Load and explore the Titanic dataset
Clean and preprocess the data
Handle missing values, outliers, and encode categorical variables
Perform feature engineering to improve model performance

Model Building and Evaluation
Select an appropriate algorithm (for example, Random Forest, XGBoost, Logistic Regression)
Train and test the model
Evaluate model using accuracy, precision, recall, F1-score, and ROC-AUC

Implement Model Version Control
Use MLflow or DVC to track model versions and data changes
Log parameters, metrics, and model artifacts
Ensure versioning practices are followed

Create a CI/CD Pipeline
Automate training, testing, and deployment steps using Python-based scripts
Use tools such as GitPython, MLflow, or others for workflow automation
Set up triggers to re-run the pipeline on new data or code updates

Deploy the Model as a REST API
Use FastAPI to deploy the trained model
Configure the API to accept patient input data and return predictions
Ensure API is testable and secure

Model Monitoring with Python-based Tools
Set up Evidently AI for monitoring model performance and data drift
Automate report generation to highlight metrics, drift, and stability
Configure alerts or reports for significant changes in model behavior

Documentation and Best Practices
Create a detailed README outlining:
Project setup, Training and deployment steps, CI/CD pipeline configuration, and Monitoring strategy
Document all tools and configurations used
Include best practices for reproducibility and scalability
Result: 

The final deliverables will include:

Codebase:

Clean, version-controlled Python scripts for preprocessing, modeling, and deployment
API Deployment:

A working FastAPI endpoint serving predictions
CI/CD System:

Automated pipeline for retraining, testing, and deployment
 Monitoring Artifacts:

Interactive reports from Evidently AI showcasing performance and data drift 
Documentation Report:

Project objectives and significance
Pipeline architecture and tool integration
Examples and screenshots of outputs
Lessons learned and future recommendations
 



