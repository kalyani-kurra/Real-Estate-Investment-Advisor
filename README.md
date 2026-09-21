# Real Estate Investment Advisor

A Machine Learning based Real Estate Investment Advisor that predicts whether a property is a good investment and estimates its future price using classification and regression models.

## Project Overview

The objective of this project is to apply Machine Learning techniques to real estate data and provide useful property investment predictions.

The project performs two main tasks:

1. **Investment Classification** – Predicts whether a property is a good investment.
2. **Future Price Prediction** – Estimates the property price after 5 years.

The trained models are integrated into an interactive Streamlit web application.

## Dataset

The project uses an India housing prices dataset containing property-related information.

### Main Features

* State
* City
* Locality
* Property Type
* BHK
* Size in SqFt
* Price in Lakhs
* Price per SqFt
* Year Built
* Furnished Status
* Floor Number
* Total Floors
* Age of Property
* Nearby Schools
* Nearby Hospitals
* Public Transport Accessibility
* Parking Space
* Security
* Amenities
* Facing
* Owner Type
* Availability Status

## Machine Learning Models

### Classification

The classification model predicts whether the property is a **Good Investment**.

**Model:** XGBoost Classifier

### Regression

The regression model predicts the estimated **Future Price after 5 Years**.

**Model:** XGBoost Regressor

## Project Workflow

```text
Dataset
    ↓
Data Preprocessing
    ↓
Feature Engineering
    ↓
Train-Test Split
    ↓
Classification Model
    ↓
Regression Model
    ↓
Model Evaluation
    ↓
Save Trained Models
    ↓
Streamlit Application
    ↓
Property Prediction
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib
* Streamlit

## Repository Structure

```text
Real-Estate-Investment-Advisor/
│
├── app.py
├── classification_model.pkl
├── regression_model.pkl
├── requirements.txt
├── README.md
├── ML_project.ipynb
└── Streamlit.ipynb
```

## Streamlit Application

The Streamlit application allows users to enter property details such as location, property type, BHK, size, price, age, amenities and other property characteristics.

After entering the details, the application provides:

* Investment classification
* Estimated future property price after 5 years

## How to Run the Project

### Step 1: Install the requirements

```bash
pip install -r requirements.txt
```

### Step 2: Run Streamlit

```bash
streamlit run app.py
```

### Step 3: Open the Application

Streamlit will provide a local URL in the terminal. Open that URL in a web browser to use the application.

## Deployment

The project can be deployed using **Streamlit Community Cloud** by connecting this GitHub repository and selecting `app.py` as the main application file.

## Project Purpose

This project demonstrates the use of Machine Learning for real estate investment analysis by combining classification and regression techniques with an interactive web application.


