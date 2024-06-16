# SVR

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

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1, 1))

# %% Fittting a  Regression to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)
print(regressor)

# %% Predict a new results with Regression
y_pred =  sc_y.inverse_transform([regressor.predict(sc_x.transform(np.array([[6.5]])))])

# %% Visualizing the SVR regression results  
plt.scatter(x, y, color='red')
# plt.plot(x, y_pred_poly)
plt.plot(x, regressor.predict(x), color='blue')
plt.title("Regression model")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# NOTE: The salary of the CEO is considered an outlier by the SVR

# %% Visualizing the polynomial regression results  
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color='red')
# plt.plot(x, y_pred_poly)
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)), color='blue')
plt.title("Regression model")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
