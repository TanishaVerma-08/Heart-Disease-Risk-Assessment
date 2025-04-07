
# ❤️ Heart Disease Risk Assessment

This project utilizes machine learning techniques to assess the risk of heart disease based on various health parameters. It includes data preprocessing, model training, and a Flask web application for user interaction.

## 📋 Project Overview

The objective of this project is to:

- Analyze health data to identify patterns associated with heart disease.
- Develop a predictive model using machine learning algorithms.
- Deploy the model in a user-friendly web application for risk assessment.

## 🛠️ Features

- **Data Analysis**: Exploratory data analysis to understand the dataset.
- **Model Training**: Implementation of machine learning models, including Random Forest.
- **Web Application**: Flask-based interface for users to input data and receive risk assessments.

## 🗂️ Project Structure

```
Heart-Disease-Risk-Assessment/
├── Heart Disease Risk Assessment.py  # Script for data analysis and model training
├── Random_forest_model.pkl           # Saved Random Forest model
├── app.py                            # Flask application script
├── heart.csv                         # Dataset used for analysis
└── README.md                         # Project documentation
```

## 🔧 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/TanishaVerma-08/Heart-Disease-Risk-Assessment.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd Heart-Disease-Risk-Assessment
   ```

3. **Install the required packages**:

   Ensure you have Python installed. Then, install the necessary libraries:

   ```bash
   pip install pandas numpy scikit-learn flask
   ```

## 🚀 Usage

1. **Data Analysis and Model Training**:

   Run the `Heart Disease Risk Assessment.py` script to perform data analysis and train the model:

   ```bash
   python "Heart Disease Risk Assessment.py"
   ```

   This script will generate a `Random_forest_model.pkl` file containing the trained model.

2. **Launching the Web Application**:

   Start the Flask application to interact with the model:

   ```bash
   python app.py
   ```

   Access the application by navigating to `http://127.0.0.1:5000/` in your web browser.

## 📝 Dataset

The dataset (`heart.csv`) contains various health parameters used to predict heart disease risk. Ensure this file is present in the project directory before running the scripts.
