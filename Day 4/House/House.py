# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

# Step 1: Data Loading and Initial Inspection
# Load the dataset
df = pd.read_csv('kc_house_data.csv')

# Display the first few rows of the dataset
print(df.head())

# Get basic information about the dataset
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Step 2: Data Cleaning
# Drop unnecessary columns (if any)
df = df.drop(['id', 'date'], axis=1)

# Handle missing values (if any)
df = df.fillna(df.median())

# Step 3: Data Wrangling
# Convert categorical variables to numerical (if any)
# For example, if 'zipcode' is categorical, we can one-hot encode it
df = pd.get_dummies(df, columns=['zipcode'], drop_first=True)

# Step 4: Exploratory Data Analysis (EDA)
# Summary statistics
print(df.describe())

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Pairplot for selected features
sns.pairplot(df[['price', 'sqft_living', 'bedrooms', 'bathrooms']])
plt.show()

# Step 5: Feature Engineering
# Create new features if necessary
df['age'] = 2023 - df['yr_built']
df['renovated'] = np.where(df['yr_renovated'] == 0, 0, 1)

# Drop original columns that are no longer needed
df = df.drop(['yr_built', 'yr_renovated'], axis=1)

# Step 6: Model Building
# Split the data into training and testing sets
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)

# Decision Tree Regression
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_test)

# Logistic Regression (if applicable)
# For classification tasks, you would need a binary target variable
# Here, we assume 'price' is continuous, so logistic regression is not applicable
# If you have a classification problem, you can use LogisticRegression from sklearn.linear_model

# Step 7: Model Evaluation
# Linear Regression Evaluation
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)
print(f"Linear Regression - MSE: {mse_lin}, R2: {r2_lin}")

# Polynomial Regression Evaluation
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print(f"Polynomial Regression - MSE: {mse_poly}, R2: {r2_poly}")

# Decision Tree Regression Evaluation
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)
print(f"Decision Tree Regression - MSE: {mse_tree}, R2: {r2_tree}")

# Step 8: Conclusion
# Compare the performance of the models and choose the best one
# For example, if Polynomial Regression has the highest R2 score, it might be the best model

# Visualize the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lin, color='blue', label='Linear Regression')
plt.scatter(y_test, y_pred_poly, color='red', label='Polynomial Regression')
plt.scatter(y_test, y_pred_tree, color='green', label='Decision Tree Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()