# Data preprocessing template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Spliting the dataset into train and test set
from sklearn.model_selection import train_test_split
# random_state is the seed
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature scaling
# from sklearn.preprocessing import StandardScaler
# standard_scaler_x = StandardScaler()
# standard_scaler_x.fit(x)
# x_train = standard_scaler_x.transform(x_train)
# x_test = standard_scaler_x.transform(x_test)

