import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('insurance.csv')

# Convert categorical variables to numeric
df['sex'] = df['sex'].map({'male': 0, 'female': 1})  # Binary encoding
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})  # Binary encoding
df = pd.get_dummies(df, columns=['region'], drop_first=True)  # One-hot encode 'region'

# Convert categorical columns to int to avoid dtype issues
df = df.astype({'region_northwest': int, 'region_southeast': int, 'region_southwest': int})

# Feature Engineering: Add Interaction Terms and Polynomial Features
df['bmi_smoker'] = df['bmi'] * df['smoker']  # Interaction: Higher risk for smoking + high BMI
df['bmi_squared'] = df['bmi'] ** 2  # Quadratic term to capture curvature
df['age_squared'] = df['age'] ** 2  # Quadratic term for age
df['children_squared'] = df['children'] ** 2  # To model increased cost with more children

# Standardize continuous features
scaler = StandardScaler()
numeric_features = ['age', 'bmi', 'children', 'bmi_smoker', 'bmi_squared', 'age_squared', 'children_squared']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Normalize the target variable (charges) to prevent instability
y_mean = df['charges'].mean()
y_std = df['charges'].std()
y = (df['charges'] - y_mean) / y_std  # Standardized target variable

# Extract features
X = df.drop(columns=['charges']).values.astype(float)  # Features (float64)

# Add bias term (column of ones)
X = np.c_[np.ones(X.shape[0]), X]

# Shuffle data and split into training (2/3) and validation (1/3)
np.random.seed(0)
indices = np.random.permutation(len(X))
train_size = int(2/3 * len(X))

X_train, X_val = X[indices[:train_size]], X[indices[train_size:]]
y_train, y_val = y[indices[:train_size]], y[indices[train_size:]]

# Compute closed-form solution using the normal equation
theta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train  # Using pseudo-inverse for stability

# Make predictions (still in normalized form)
y_train_pred = X_train @ theta
y_val_pred = X_val @ theta

# Reverse normalization of target variable
y_train_pred = y_train_pred * y_std + y_mean
y_val_pred = y_val_pred * y_std + y_mean
y_train = y_train * y_std + y_mean  # Reverse actual y_train
y_val = y_val * y_std + y_mean  # Reverse actual y_val

# Compute RMSE
rmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
rmse_val = np.sqrt(np.mean((y_val - y_val_pred) ** 2))

# Corrected SMAPE function (ensures denominator is |Y| + |Yhat|)
def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100  # Fix denominator

smape_train = smape(y_train, y_train_pred)
smape_val = smape(y_val, y_val_pred)

# Print results
print(f"Training RMSE: {rmse_train:.2f}")
print(f"Validation RMSE: {rmse_val:.2f}")
print(f"Training SMAPE: {smape_train:.2f}%")
print(f"Validation SMAPE: {smape_val:.2f}%")
