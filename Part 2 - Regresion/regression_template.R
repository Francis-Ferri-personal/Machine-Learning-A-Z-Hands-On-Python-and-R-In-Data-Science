# Regression template

# Import dataset
dataset = read.csv("Position_Salaries.csv")
dataset = dataset[,2:3]

# install.packages('caTools')

# Splitting the dataset into Training set and Test set
# library(caTools)
# set.seed(123)
# # Split ratio is for training set in this case not for test
# # The column used is y
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split==TRUE)
# test_set = subset(dataset, split==FALSE)

# Feature scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])

# Fitting the Regression to the dataset
# Create regressor here

# Predicting a new result with regression
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualizing the Regression model results
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") + 
  geom_line(aes(x= dataset$Level, y = predict(regressor, newdata = dataset)), color ='blue' ) +
  ggtitle("Truth or Bluff (Regression Model)") +
  xlab('Level') + 
  ylab("Salary")

# Vis
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") + 
  geom_line(aes(x= x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), color ='blue' ) +
  ggtitle("Truth or Bluff (Regression Model)") +
  xlab('Level') + 
  ylab("Salary")

