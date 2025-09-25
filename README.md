# Linear Regression Model for House Price Prediction 🏡
This repository contains the code and dataset for a machine learning project focused on predicting house prices using a linear regression model. The project was completed as part of a task to demonstrate an understanding of regression modeling, data preprocessing, and model evaluation. 

# Project Objective
The main goal of this project is to build a predictive model that can accurately estimate house prices based on various features such as area, number of bedrooms, and furnishing status. We use a multiple linear regression model to analyze the relationships between these features and the target variable, price.

# Dataset
The analysis is based on the Housing.csv dataset, which includes a comprehensive set of house-related information:
price: The sale price of the house (target variable).
area: The size of the house in square feet.
bedrooms: Number of bedrooms.
bathrooms: Number of bathrooms.
stories: Number of stories in the house.
mainroad: Whether the house is on a main road (yes/no).
guestroom: Whether the house has a guest room (yes/no).
basement: Whether the house has a basement (yes/no).
hotwaterheating: Whether the house has hot water heating (yes/no).
airconditioning: Whether the house has air conditioning (yes/no).
parking: Number of parking spaces.
prefarea: Whether the house is in a preferred area (yes/no).
furnishingstatus: The furnishing status of the house (furnished, semi-furnished, or unfurnished).

# Technologies and Libraries
The project is built using Python and the following key libraries:
Pandas: For efficient data loading, cleaning, and manipulation.
Scikit-learn: For implementing the linear regression model, splitting data, and evaluating performance.
Matplotlib & Seaborn: For data visualization, specifically for plotting the predicted vs. actual prices.

# Model Evaluation and Results
After running the model, the following evaluation metrics were obtained:
Mean Absolute Error (MAE): Measures the average magnitude of the errors in a set of predictions, without considering their direction.
Mean Squared Error (MSE): The average of the squared errors. It gives more weight to larger errors.
R-squared (R2) Score: A crucial metric that indicates the proportion of the variance in the dependent variable (price) that is predictable from the independent variables. Our model achieved an R 2 score of 0.6529, which means it explains approximately 65.3% of the variance in house prices. This is considered a good result for a complex real-world problem like house price prediction.



