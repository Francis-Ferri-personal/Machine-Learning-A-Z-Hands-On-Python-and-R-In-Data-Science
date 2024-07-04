# Hierarchical clustering

# Importing the mall dataset
dataset <- read.csv("Mall_Customers.csv")
x = dataset[4:5]

# Using the dendrogram to find the optimal number of cluster
dendrogram = hclust(dist(x, method = 'euclidean'), method = "ward.D2")
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Çustomers',
     ylab = 'Euclidean distance')

# Fitting the hierarchical clustering to the mall dataset
hc = hclust(dist(x, method = 'euclidean'), method = "ward.D2")
y_hc = cutree(hc, 5)

# Visualizing the clusters
library(cluster)
clusplot(x, y_hc, lines=0, shade = TRUE, color = TRUE,
         labels = 2, plotchar = FALSE,
         span=TRUE,
         main = paste('Cluster of clients'),
         xlab= "Annual ncome",
         ylab="Spending score")
