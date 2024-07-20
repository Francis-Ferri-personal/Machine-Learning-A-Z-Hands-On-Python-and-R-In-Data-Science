#%% Grid Search
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
# random_state is the seed
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
standard_scaler_x = StandardScaler()
standard_scaler_x.fit(x)
x_train = standard_scaler_x.transform(x_train)
x_test = standard_scaler_x.transform(x_test)

# Fitting the classifier to the training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0) # gamma is important to get better results
classifier.fit(x_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(x_test)
 
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier, X = x_train, y = y_train, cv=10) # n_jobs = -1 if yu want to se all the cpu
print("Mean:", accuracies.mean())
print("Std:", accuracies.std())

# %% Applying Grid Search to find the best model ad the best parameters
from sklearn.model_selection import GridSearchCV

parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001]} # 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
]

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10,  n_jobs = -1) # cv is for k-fold cross validation, n_jobs = -1 for using all the power available

grid_search = grid_search.fit(x_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


#%% Visualizing the Training set results
from matplotlib.colors import ListedColormap
x_set, y_set = standard_scaler_x.inverse_transform(x_train), y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 10, stop = x_set[:, 0].max() + 10, step = 0.5),
                     np.arange(start = x_set[:, 1].min() - 1000, stop = x_set[:, 1].max() + 1000, step = 0.5))
plt.contourf(x1, x2, classifier.predict(standard_scaler_x.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualizing the Test set results
from matplotlib.colors import ListedColormap
x_set, y_set = standard_scaler_x.inverse_transform(x_test), y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 10, stop = x_set[:, 0].max() + 10, step = 0.5),
                     np.arange(start = x_set[:, 1].min() - 1000, stop = x_set[:, 1].max() + 1000, step = 0.5))
plt.contourf(x1, x2, classifier.predict(standard_scaler_x.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Testing set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()