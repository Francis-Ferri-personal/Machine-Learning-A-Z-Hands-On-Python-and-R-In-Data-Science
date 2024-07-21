#%% XGBoost

# Install XGBoost
# pip install xgboost

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

# %% Fitting XGBoost to the training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

# %% Part three making predictions and evaluating the model

# Predicting the test set results
y_pred = classifier.predict(x_test)
# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# %% Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier, X = x_train, y = y_train, cv=10) # n_jobs = -1 if yu want to se all the cpu
print("Mean:", accuracies.mean())
print("Std:", accuracies.std())
