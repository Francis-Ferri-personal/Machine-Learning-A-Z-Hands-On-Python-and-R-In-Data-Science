# PCA

# Import dataset
dataset = read.csv("Wine.csv")

# install.packages('caTools')

# Spliting the datasetset into Training set and Test set
library(caTools)
set.seed(123)
# Split ratio is for training set in this case not for test
# The column used is y
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

# Feature Scaling
training_set[-14] = scale(training_set[-14])
test_set[-14] = scale(test_set[-14])

# Applying PCA
# install.packages('caret')
library(caret)

# install.packages('e1071')
library(e1071)

pca = preProcess(x = training_set[-14], method = 'pca', pcaComp = 2)
training_set = predict(pca, training_set)
training_set =  training_set[c(2, 3, 1)]

test_set = predict(pca, test_set)
test_set =  test_set[c(2, 3, 1)]

# Fitting Logistic Regression to the training set
classifier = svm(formula = Customer_Segment ~ ., data = training_set, type = 'C-classification', kernel='linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the confusion matrix
cm = table(test_set[,3], y_pred)

# Visualizing the Training set results
# install.packages("devtools")
# devtools::install_version("ElemStatLearn", version = "2015.6.26.2", repos = "http://cran.us.r-project.org")
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'dodgerblue', ifelse(y_grid == 1, 'deepskyblue', 'salmon')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'dodgerblue3', ifelse(set[, 3] == 1, 'deepskyblue', 'salmon')))

# Visualizing the Training set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'dodgerblue', ifelse(y_grid == 1, 'deepskyblue', 'salmon')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'dodgerblue3', ifelse(set[, 3] == 1, 'deepskyblue', 'salmon')))