#%% Hierarchical clustering
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the mall dataset with pandas
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, 3:5].values

#%% Using the dendrogram to find the optimal number of clusters
# ward method tries to minimize the variance within the clusters
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distances")
plt.show()

#%% Fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(x)

#%% Visualizing teh clusters
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], c='red', label="Careful")
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], c='blue', label="Standard")
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], c='green', label="Target")
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], c='cyan', label="Careless")
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], c='magenta', label="Sensible")
plt.title('CLuster of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()