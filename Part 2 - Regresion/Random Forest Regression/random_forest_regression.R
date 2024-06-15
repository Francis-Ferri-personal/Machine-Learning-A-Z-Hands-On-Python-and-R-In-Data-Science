# Random Forest Regression Regression

# Import dataset
dataset = read.csv("Position_Salaries.csv")
dataset = dataset[,2:3]

# Fitting the Regression to the dataset
# installed.packages("randomForest")
library(randomForest)
set.seed(1234)
# dataset[1} # Dataframe
# dataset$Salary # vector
regressor = randomForest(x = dataset[1], y = dataset$Salary, ntree = 500)
# formula = Salary ~ ., data = dataset, control = rpart.control(minsplit = 1)


# Predicting a new result with regression
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualizing the Regression model results in high resolution
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") + 
  geom_line(aes(x= x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), color ='blue' ) +
  ggtitle("Truth or Bluff (Random Forest Regression Model)") +
  xlab('Level') + 
  ylab("Salary")

