
# Import dataset
dataset = read.csv("Data.csv")
datste = dataset[,2:3]
# Taking care of missing data\


# install.packages('caTools')

# Spliting the datasetset into Training set and Test set
library(caTools)
set.seed(123)
# Split ratio is for training set in this case not for test
# The column used is y
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

# Feature scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])