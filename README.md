# House Price Prediction using Linear Regression

This Python code builds and evaluates a linear regression model to predict house prices, performing data preprocessing, training, and assessment of the model's performance.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-blueviolet?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![Pandas](https://img.shields.io/badge/Pandas-2C2D72?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-green?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-orange?logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)

## Overview 🏡

This repository contains a machine learning project focused on predicting house prices using a Linear Regression model. The project demonstrates a complete workflow from data loading and preprocessing to model training, evaluation, and saving the trained pipeline.

## Features ✨

* **Data Loading & Exploration:** Loads a CSV dataset (`train.csv`) and performs initial data inspection (head, info, describe, missing values).
* **Dynamic Feature Handling:** Automatically identifies numerical and categorical features for appropriate preprocessing.
* **Robust Preprocessing Pipeline:**
    * **Missing Value Imputation:** Fills numerical missing values with the median and categorical missing values with the mode.
    * **Numerical Scaling:** Applies `StandardScaler` to numerical features.
    * **Categorical Encoding:** Uses `OneHotEncoder` for categorical features to convert them into a format suitable for the model.
* **Linear Regression Model:** Implements `LinearRegression` for price prediction.
* **Scikit-learn Pipeline:** Utilizes `Pipeline` and `ColumnTransformer` for an organized and reproducible workflow, ensuring proper data transformation before model training.
* **Model Evaluation:** Calculates and displays Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) score.
* **Prediction Visualization:** Generates a scatter plot comparing actual vs. predicted prices.
* **Feature Importance:** Displays the coefficients of the linear regression model as an indicator of feature importance.
* **Model Persistence:** Saves the entire trained pipeline (including preprocessors and the model) to a `.pkl` file for future use without retraining.

## Installation and Setup 🚀

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YourUsername]/[YourRepositoryName].git
    cd [YourRepositoryName]
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install the required libraries:**
    ```bash
    pip install pandas scikit-learn numpy matplotlib seaborn joblib
    ```
    *Alternatively, you can create a `requirements.txt` file:*
    ```bash
    # requirements.txt
    pandas
    scikit-learn
    numpy
    matplotlib
    seaborn
    joblib
    ```
    *Then install with:*
    ```bash
    pip install -r requirements.txt
    ```

## Usage 📈

### Training and Evaluating the Model

1.  Place your dataset named `train.csv` in the root directory of the project.
2.  Run the main Python script:
    ```bash
    python House_Price_Prediction.py
    ```
    This script will:
    * Load the `train.csv` dataset.
    * Perform data preprocessing (imputation, scaling, one-hot encoding).
    * Train the Linear Regression model.
    * Print evaluation metrics (MSE, RMSE, R2 Score).
    * Display a plot of actual vs. predicted prices.
    * Print feature coefficients.
    * Save the trained model pipeline as `house_price_prediction_model.pkl`.

### Using the Pre-trained Model for Predictions 🔮

You can load the `house_price_prediction_model.pkl` file and use it to make predictions on new data without re-training.

```python
import pandas as pd
import joblib

# Load the trained pipeline
pipeline = joblib.load('house_price_prediction_model.pkl')

# Example new data (replace with your actual new data structure)
# Ensure new_data has the same columns and data types as your training data (except 'SalePrice'/'price')
new_data = pd.DataFrame({
    # Example: 'LotArea': [8450], 'OverallQual': [7], ...
    # Make sure to include all features present in your original X_train, even if they have default values.
    # For demonstration, let's assume 'train.csv' had columns like this:
    'MSSubClass': [60], 'MSZoning': ['RL'], 'LotFrontage': [65.0], 'LotArea': [8450],
    'Street': ['Pave'], 'Alley': [None], 'LotShape': ['Reg'], 'LandContour': ['Lvl'],
    'Utilities': ['AllPub'], 'LotConfig': ['Inside'], 'LandSlope': ['Gtl'],
    'Neighborhood': ['CollgCr'], 'Condition1': ['Norm'], 'Condition2': ['Norm'],
    'BldgType': ['1Fam'], 'HouseStyle': ['2Story'], 'OverallQual': [7],
    'OverallCond': [5], 'YearBuilt': [2003], 'YearRemodAdd': [2003],
    'RoofStyle': ['Gable'], 'RoofMatl': ['CompShg'], 'Exterior1st': ['VinylSd'],
    'Exterior2nd': ['VinylSd'], 'MasVnrType': ['BrkFace'], 'MasVnrArea': [196.0],
    'ExterQual': ['Gd'], 'ExterCond': ['TA'], 'Foundation': ['PConc'],
    'BsmtQual': ['Gd'], 'BsmtCond': ['TA'], 'BsmtExposure': ['No'],
    'BsmtFinType1': ['GLQ'], 'BsmtFinSF1': [706], 'BsmtFinType2': ['Unf'],
    'BsmtFinSF2': [0], 'BsmtUnfSF': [150], 'TotalBsmtSF': [856],
    'Heating': ['GasA'], 'HeatingQC': ['Ex'], 'CentralAir': ['Y'],
    'Electrical': ['SBrkr'], '1stFlrSF': [856], '2ndFlrSF': [961],
    'LowQualFinSF': [0], 'GrLivArea': [1788], 'BsmtFullBath': [1],
    'BsmtHalfBath': [0], 'FullBath': [2], 'HalfBath': [1], 'BedroomAbvGr': [3],
    'KitchenAbvGr': [1], 'KitchenQual': ['Gd'], 'TotRmsAbvGrd': [8],
    'Functional': ['Typ'], 'Fireplaces': [0], 'FireplaceQu': [None],
    'GarageType': ['Attchd'], 'GarageYrBlt': [2003.0], 'GarageFinish': ['RFn'],
    'GarageCars': [2], 'GarageArea': [548], 'GarageQual': ['TA'],
    'GarageCond': ['TA'], 'PavedDrive': ['Y'], 'WoodDeckSF': [0],
    'OpenPorchSF': [61], 'EnclosedPorch': [0], '3SsnPorch': [0],
    'ScreenPorch': [0], 'PoolArea': [0], 'PoolQC': [None], 'Fence': [None],
    'MiscFeature': [None], 'MiscVal': [0], 'MoSold': [2], 'YrSold': [2008],
    'SaleType': ['WD'], 'SaleCondition': ['Normal'], 'Id': [1] # Id column might be present in train.csv
})

# Make predictions
predicted_price = pipeline.predict(new_data)
print(f"Predicted House Price: ${predicted_price[0]:,.2f}")
