#%% Artificial Neural NEtwork

# Installing Theano
# pip install Theano

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# %% 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% Importing dataset
dataset = pd.read_csv("Churn_Modelling.csv")

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# # taking care of missing data
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# imputer = imputer.fit(x[:, 1:3])
# x[:, 1:3] = imputer.transform(x[:, 1:3])

#%% Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])

labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = ct.fit_transform(x)
x = x[:, 1:]

# %% # Spiting the dataset into train and test set
from sklearn.model_selection import train_test_split
# random_state is the seed
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#%% Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# %% Part 2 - Make the ANN
import keras
from keras import Sequential
from keras.api.layers import Dense

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(6,  kernel_initializer="glorot_uniform", activation='relu')) # Not necessary to add the input dims
classifier.add(Dense(6,  kernel_initializer="glorot_uniform", activation='relu'))
classifier.add(Dense(1,  kernel_initializer="glorot_uniform", activation='sigmoid'))
 
# %% Compiling the ANN
classifier.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(x_train, y_train, batch_size=10, epochs=100)

# %% Part three making predictions and evaluating the model

# Predicting the test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)