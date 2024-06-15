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

# %% Preprocessing

# %% Fittting a  Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
 
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

# %% Predict a new results with Regression
regressor.predict([[6.5]])

# %% Visualizing the polynomial regression results  
# NOTE: just plotting the values of the data will show us a very good plot, but it is a trap
# probably we weill not notice the starirs patterns. Use the code for plotting many predictions instead
# plt.scatter(x, y, color='red')
# plt.plot(x, y_pred_poly)
# plt.plot(x, regressor.predict(x), color='blue')
# plt.title("Regression model (Decision Tree Regression)")
# plt.xlabel("Position level")
# plt.ylabel("Salary")
# plt.show()

# %% Visualizing the regression results  
x_grid = np.arange(x[0], x[-1], 0.01)
x_grid = np.reshape(x_grid, (len(x_grid), 1))
plt.scatter(x, y, color='red')
# plt.plot(x, y_pred_poly)
plt.plot(x_grid, regressor.predict(x_grid))
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.title("Polynomial Linear regression (Decision Tree Regression)")
plt.show()


