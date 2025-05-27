import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings

# Load the dataset
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: 'train.csv' not found.")
    raise
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    raise


# --- Data Exploration and Preprocessing ---
print("First few rows of the dataset:")
print(df.head())
print("\nDataset information:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# Handle potential missing values (checking and filling/dropping if needed)
print("\nMissing values per column:")
print(df.isnull().sum())

# Separate target variable and features
if 'SalePrice' in df.columns:
    target_variable = 'SalePrice'
elif 'price' in df.columns:
    target_variable = 'price'
else:
    print("Error: Target variable 'SalePrice' or 'price' not found in the dataset. Please ensure the dataset includes the target variable.")
    raise KeyError("Target variable not found")  # Stop if target variable not found

y = df[target_variable]
X = df.drop(columns=[target_variable], axis=1, errors='ignore') # drop target variable


# Identify categorical and numerical features dynamically
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()


# --- Data Preprocessing ---

# Impute missing values
for col in numerical_features:
    X[col] = X[col].fillna(X[col].median())
for col in categorical_features:
    X[col] = X[col].fillna(X[col].mode()[0])

# --- Create a Preprocessing Pipeline ---
# Create transformers for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply transformations to the correct columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# --- Create and Train the Linear Regression Model ---
# Create the model
model = LinearRegression()

# Create a pipeline that first preprocesses the data and then trains the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using the pipeline
with warnings.catch_warnings():  # Suppress warnings during training
    warnings.filterwarnings("ignore")
    pipeline.fit(X_train, y_train)

# --- Make Predictions and Evaluate the Model ---
# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R2) Score: {r2:.2f}')

# --- Visualize Predictions ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# --- Feature Importance (for Linear Regression, can be less directly interpretable with one-hot encoding) ---
if isinstance(pipeline.named_steps['regressor'], LinearRegression):
    try:
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        coefficients = pipeline.named_steps['regressor'].coef_

        feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
        print("\n--- Feature Importance (Coefficients) ---")
        print(feature_importance)
    except AttributeError:
        print("\nFeature importance is not available because the model is not a linear regression.")
else:
    print("\nFeature importance is not directly available for this type of model.")

# --- Saving the Trained Model (Optional) ---
# Save the trained pipeline to a file
joblib.dump(pipeline, 'house_price_prediction_model.pkl')
print("\nTrained model saved as 'house_price_prediction_model.pkl'")