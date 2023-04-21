# Data preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_x = LabelEncoder()
x[:, 0] = labelEncoder_x.fit_transform(x[:, 0])
ct = ColumnTransformer([("one_hot_encoder", OneHotEncoder(), [0])], remainder = 'passthrough')
x = ct.fit_transform(x)

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Spliting the dataset into train and test set
from sklearn.model_selection import train_test_split
# random_state is the seed
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
standard_scaler_x = StandardScaler()
standard_scaler_x.fit(x)
x_train = standard_scaler_x.transform(x_train)
x_test = standard_scaler_x.transform(x_test)

