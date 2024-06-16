# Data preprocessing

# %% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% import dataset
dataset = pd.read_csv("Position_Salaries.csv")
# x = dataset.iloc[:, 1].values # size: (10)
x = dataset.iloc[:, 1:2].values # size: (10, 1) Do this way to keep your data as a matrix
y = dataset.iloc[:, 2].values

# %% Fittting a Linear Regression to the dataset

from sklearn.linear_model import LinearRegression

lin_regressor = LinearRegression()

lin_regressor.fit(x, y)

y_pred = lin_regressor.predict(x)


# %% Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

poly_lin_pred = LinearRegression()
poly_lin_pred.fit(x_poly, y)

y_pred_poly = poly_lin_pred.predict(x_poly)

# %% Visualizing the linear regression results
plt.scatter(x, y, color='red')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.title("Linear regression")
plt.plot(x, y_pred)
plt.show()

# %% Visualizing the polynomial regression results  
plt.scatter(x, y, color='red')
# plt.plot(x, y_pred_poly)
plt.plot(x, poly_lin_pred.predict(poly_reg.fit_transform(x)))
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.title("Polynomial Linear regression")
plt.show()

# %%
x_grid = np.arange(x[0], x[-1], 0.1)
x_grid = np.reshape(x_grid, (len(x_grid), 1))

# %% Visualizing the polynomial regression results  
plt.scatter(x, y, color='red')
# plt.plot(x, y_pred_poly)
plt.plot(x_grid, poly_lin_pred.predict(poly_reg.fit_transform(x_grid)))
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.title("Polynomial Linear regression")
plt.show()

# %% Predict a new results with Linear Regression
lin_regressor.predict([[6.5]])

# %% Predicting a new resut with Polynomial Regression
poly_lin_pred.predict(poly_reg.fit_transform([[6.5]]))


