# Data preprocessing template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Spliting the dataset into train and test set
from sklearn.model_selection import train_test_split
# random_state is the seed
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Feature scaling
# from sklearn.preprocessing import StandardScaler
# standard_scaler_x = StandardScaler()
# standard_scaler_x.fit(x)
# x_train = standard_scaler_x.transform(x_train)
# x_test = standard_scaler_x.transform(x_test)

# Fitting Simple linear regresor to the training set
from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
model = regresor.fit(x_train, y_train)

# Prediction of the test results
y_pred = regresor.predict(x_test)

# Visualizing the training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regresor.predict(x_train), color='blue')
plt.title("Experience VS Salary (Training)")
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()

# Visualizing the training set results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regresor.predict(x_train), color='blue') 
# Como es linear regresion y el mismo modelo no es necesario cambiar
plt.title("Experience VS Salary (Test)")
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()