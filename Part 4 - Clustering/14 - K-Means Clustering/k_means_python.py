#%% Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Importing the mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3, 4]].values

#%% Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

wcss = []
# Inertia: Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
for i in range(10):
    kmeans = KMeans(n_clusters=i+1, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

#%% Plot the wcss
plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel("Numeber of clusters")
plt.ylabel("WCSS")
plt.show()

#%% Applying k-means to the mall dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)

#%% Visualizing the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], c='red', label="Standard")
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], c='blue', label="Target")
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], c='green', label="Careful")
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], c='cyan', label="Sensible")
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], c='magenta', label="Careless")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label="Centroids")
plt.title('CLuster of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()