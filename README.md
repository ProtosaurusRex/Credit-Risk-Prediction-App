# Credit-Risk-Prediction-App
A leightweight streamlight app I built using scikit learn prediciting credit risk
📊 Credit Risk Modeling Using Machine Learning

An end-to-end data analysis and machine learning project focused on predicting credit risk using structured financial data.
This project demonstrates practical skills in data cleaning, exploratory analysis, model training, evaluation, and deployment.

📌 Project Overview

Financial institutions rely on accurate credit risk assessment to reduce loan defaults and manage financial exposure.
In this project, I built a machine learning model that classifies loan applicants as:

Low Risk (Good Credit)

High Risk (Bad Credit)

The project covers the complete data workflow from raw data to a deployable prediction application.

🎯 Objectives

Perform exploratory data analysis on real-world financial data

Prepare and encode structured categorical and numerical features

Train and compare multiple classification models

Select the best-performing model

Deploy a simple interactive prediction app using Streamlit

🗂 Dataset

Source: German Credit Dataset (Kaggle)

Size: ~1,000 records (after cleaning)

Target Variable: Risk (Good / Bad)

Key Features

Age

Sex

Job category

Housing status

Saving accounts

Checking account

Credit amount

Loan duration

Loan purpose

🧹 Data Cleaning & Preparation

Removed unnecessary index columns

Handled missing values by dropping incomplete rows

Checked for duplicates and invalid data types

Verified class distribution for imbalance

The final dataset contained no missing or duplicate values.

📊 Exploratory Data Analysis (EDA)

EDA focused on identifying risk patterns rather than visual aesthetics.

Key Insights

Credit risk is imbalanced, with more good loans than bad

Larger credit amounts and longer loan durations tend to be higher risk

Certain housing types and loan purposes show different risk distributions

Low correlation among numeric features supports tree-based models

🛠 Feature Engineering

Selected relevant numerical and categorical features

Applied Label Encoding to categorical variables

Encoded the target variable into binary format

Saved encoders using joblib for consistent inference

🤖 Modeling Approach

Three supervised classification models were trained and compared:

Decision Tree

Random Forest

Extra Trees Classifier

Training Strategy

80/20 train-test split with stratification

Hyperparameter tuning using GridSearchCV

Class imbalance handled using class_weight='balanced'

Model Selection

The Extra Trees Classifier achieved the highest test accuracy and was selected as the final model.

📈 Model Evaluation

Primary metric: Accuracy

Important Note

In real credit risk systems, metrics such as recall, precision, and ROC-AUC are often more important than accuracy alone, as misclassifying high-risk applicants has a higher cost.

This project acknowledges these considerations while using accuracy as a baseline evaluation metric.

🚀 Deployment

A lightweight Streamlit application was built to demonstrate deployment:

User inputs applicant details

Inputs are encoded using saved encoders

Model returns a real-time risk prediction

Deployment is included as a minor extension to showcase end-to-end capability.

🧰 Tech Stack

Language: Python

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

Models: Decision Tree, Random Forest, Extra Trees

Deployment: Streamlit

Model Persistence: joblib

⚠ Limitations & Future Improvements
Current Limitations

Accuracy used instead of recall-focused metrics

No ROC-AUC or confusion matrix analysis

Label encoding instead of one-hot encoding

Small dataset size

No final cross-validation beyond GridSearchCV

Future Enhancements

Introduce recall-optimized evaluation

Add ROC-AUC and confusion matrices

Implement probability-based risk scoring

Compare additional models (Logistic Regression, Gradient Boosting)

Add fairness and bias analysis

✅ Skills Demonstrated

Data cleaning and validation

Exploratory data analysis with business interpretation

Feature engineering for structured data

Supervised machine learning model comparison

Hyperparameter tuning

Model persistence and deployment

Awareness of real-world credit risk evaluation challenges

📎 Resources

Dataset: German Credit Dataset (Kaggle)

Full project walkthrough: YouTube tutorial (linked in notebook)

⭐ If you find this project useful, feel free to star the repository!
