# Decision Tree Regression



# Import dataset
dataset = read.csv("Position_Salaries.csv")
dataset = dataset[,2:3]

# Fitting the Regression to the dataset
# install.packages("rpart")
library(rpart)
regressor = rpart(formula = Salary ~ ., data = dataset, control = rpart.control(minsplit = 1))


# Predicting a new result with regression
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualizing the Regression model results
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") + 
  geom_line(aes(x= dataset$Level, y = predict(regressor, newdata = dataset)), color ='blue' ) +
  ggtitle("Truth or Bluff (Decision Tree Regression Model)") +
  xlab('Level') + 
  ylab("Salary")

# Visualizing the Regression model results in high resolusion
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") + 
  geom_line(aes(x= x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), color ='blue' ) +
  ggtitle("Truth or Bluff (Decision Tree Regression Model)") +
  xlab('Level') + 
  ylab("Salary")

