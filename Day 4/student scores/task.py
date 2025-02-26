# Important librearires 
import pandas as pd 
from sklearn.model_selection import train_test_split # split data 
from sklearn.linear_model import LinearRegression # call model
from sklearn.metrics import r2_score # calculate accuracy
from matplotlib import pyplot as plt # visualization
import numpy as np # calculate mean to edit nan values

# Path Data
path = r'C:\Users\moham\Desktop\NTI_ETA_CU\Session4\student_scores.csv'
data = pd.read_csv(path)
print(data)

# Number of records with at least one NaN
num_records_with_nan = data.isna().any(axis=1).sum()
print(f"Number of records with at least one NaN: {num_records_with_nan}")

# Replace NaN 
data = data.fillna(np.mean(data['Hours']))
print(data)

# Featurees and split data 
x = data[['Hours']]
y = data['Scores']
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.3 , random_state=42)

# Model Linear regression 
model = LinearRegression()
model.fit(x_train , y_train)

y_pred = model.predict(x_test)

# Visualization linear regression 
plt.figure('Linear Regression')
plt.scatter(x , y , color='blue' , label='Data Points')
plt.plot(x_test , y_pred , color='red' , label='Linear Regression')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.legend()
plt.show()

# Accuracy 
r2 = r2_score(y_test , y_pred)
print(f"Accuracy linear = {r2 * 100:.2f} %")

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

# define degree
degree = 3
pol_deg = PolynomialFeatures(degree= 3)

# model 
pol_model = LinearRegression()
pol_model.fit(pol_deg.fit_transform(x_train),y_train)
y_pred_pol = pol_model.predict(pol_deg.fit_transform(x_test))

# Accuracy polynomial regression
r2 = r2_score(y_test , y_pred_pol)
print(f"Accuracy pol. = {r2 * 100:.2f} %")

# Visualization for Polynomial Regression
plt.scatter(x, y, color='blue', label='Data Points')
x_sorted, y_sorted = zip(*sorted(zip(x['Hours'], y)))  # Sort for a smooth curve
plt.plot(x_sorted, pol_model.predict(pol_deg.fit_transform(pd.DataFrame(x_sorted))), color='green',
    label=f'Polynomial Regression (Degree = {degree})',
)
plt.xlabel('Hours')
plt.ylabel('Score')
plt.legend()
plt.title(f"Polynomial Regression:(Degree = {degree})")
plt.show()
