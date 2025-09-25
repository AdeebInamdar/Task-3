# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load the dataset
df = pd.read_csv('Housing.csv')

# 3. Preprocess the dataset
# The objective is to predict 'price' using 'area' and other features.
# The PDF mentions simple & multiple linear regression, so we will use multiple features.
# Handle categorical variables (yes/no and furnishingstatus) using one-hot encoding.
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define the features (X) and the target variable (y)
X = df_processed.drop('price', axis=1)
y = df_processed['price']

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Fit a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model using MAE, MSE, and R-squared
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# 7. Plotting the regression line (for 'area' vs 'price')
# Since we have multiple features, we can't plot a simple 2D regression line for all features.
# Instead, we will plot the predicted vs. actual prices to visualize model performance.
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.show()

# Interpret coefficients
# The coefficients show the impact of each feature on the house price.
# For 'mainroad_yes', the coefficient indicates the change in price for a house on a main road, holding all other features constant.
print("\nModel Coefficients:")
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# For multiple linear regression, the interpretation of coefficients depends on the feature type:
# For numeric features like 'area', the coefficient represents the change in price for a one-unit increase in area.
# For binary dummy variables like 'mainroad_yes', the coefficient represents the change in price for a house on a main road (vs. not on a main road), holding all other features constant.