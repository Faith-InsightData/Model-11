# ❤️

# Heart Disease Prediction
 Welcome to the Heart Disease Prediction project! This repository contains the code and resources for predicting heart disease using machine learning. 🏊‍♂️

# 📋 Table of Contents
 Introduction
 
 Dataset
 
 Installation
 
 Usage
 
 Model Training

# 🌟 Introduction
 Heart disease is one of the leading causes of death worldwide. This project aims to predict the presence of heart disease in patients using various machine learning techniques. By analyzing patient data, we can identify patterns and risk factors associated with heart disease. 🩺

# 📊 Dataset
 The dataset used for training the model is named heart-disease.csv. This dataset contains various health-related features such as:

age: Age of the patient
sex: Gender (1 = male; 0 = female)
cp: Chest pain type (4 values)
trestbps: Resting blood pressure
chol: Serum cholesterol in mg/dl
fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
restecg: Resting electrocardiographic results
thalach: Maximum heart rate achieved
exang: Exercise-induced angina (1 = yes; 0 = no)
oldpeak: ST depression induced by exercise relative to rest
slope: The slope of the peak exercise ST segment
ca: Number of major vessels (0-3) colored by fluoroscopy
thal: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
target: Presence of heart disease (1 = disease, 0 = no disease


## 🛠️ Installation
To get started, clone this repository and install the required dependencies:

bash
Copy code
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt


## 🚀 Usage
To run the prediction model, use the following command:

#  🧠 Model Training
 The model is trained using a Decision Tree Classifier. The training script is located in 3.0-decision-tree-model-training.py. Here’s a brief overview of the steps:

# Load Dataset: Load the heart disease dataset.
 Prepare Data: Split the data into features and target variables.
 Train Model: Train the Decision Tree model on the training data.
Evaluate Model: Evaluate the model’s performance on the test data.




## 🛠️ Installation
To get started, clone this repository and install the required dependencies:


 git clone https://github.com/yourusername/heart-disease-prediction.git
  cd heart-disease-prediction
  pip install -r requirements.txt
  
## 🚀 Usage
To run the prediction model, use the following command:


python main.py
Make sure to update the file paths in the script if necessary.

## 🧠 Model Training
The model is trained using a Decision Tree Classifier. The training script is located in 3.0-decision-tree-model-training.py. Here’s a brief overview of the steps:

##  Load Dataset: Load the heart disease dataset.
Prepare Data: Split the data into features and target variables.
Train Model: Train the Decision Tree model on the training data.
Evaluate Model: Evaluate the model’s performance on the test data.

## 📊 Model Performance

 ## First Classification Results:
Metric	Precision	Recall	F1-Score	Support
No Disease (0)	0.83	0.83	0.83	29
Disease (1)	0.84	0.84	0.84	32
Accuracy: 0.84 (84%)
Macro Average: 0.84
Weighted Average: 0.84

## Second Classification Results:
Metric	Precision	Recall	F1-Score	Support
No Disease (0)	0.69	0.86	0.77	29
Disease (1)	0.84	0.66	0.74	32
Accuracy: 0.75 (75%)
Macro Average: 0.77
Weighted Average: 0.75

## ✅ Accuracy Scores
First Classification Accuracy: 83.6%
Second Classification Accuracy: 75.4%

## 🚀 Next Steps
Improve model accuracy with feature selection and hyperparameter tuning.
Expand the dataset for more robust results.
Build an interactive front-end using Streamlit.84 (84%)
