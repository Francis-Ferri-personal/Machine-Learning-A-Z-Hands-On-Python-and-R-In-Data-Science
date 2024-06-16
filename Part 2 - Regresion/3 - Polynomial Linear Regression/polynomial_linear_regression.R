# Polynomial Regression


# Import dataset
dataset = read.csv("Position_Salaries.csv")
dataset = dataset[,2:3]
# Taking care of missing data\


# install.packages('caTools')

# Spliting the datasetset into Training set and Test set
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

# Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ ., data = dataset)
summary(lin_reg)

# Fitting Polynomial Regression to the dataset
# To make the polynomials we have to add the columns manually
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4

poly_reg = lm(formula = Salary ~ ., data = dataset, )
summary(poly_reg)

# Visualizing th Linear Regression results
# install.packages("ggplot2")
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") + 
  geom_line(aes(x= dataset$Level, y = predict(lin_reg, newdata = dataset)), color ='blue' ) +
  ggtitle("Truth or Bluff (Linear Regression)") +
  xlab('Level') + 
  ylab("Salary")
  

# Visualizing the Polynomial regression results
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") + 
  geom_line(aes(x= dataset$Level, y = predict(poly_reg, newdata = dataset)), color ='blue' ) +
  ggtitle("Truth or Bluff (Polynomial Regression)") +
  xlab('Level') + 
  ylab("Salary")

# Predicting a new result with Linear Regression
y_pred = predict(lin_reg, data.frame(Level = 6.5))

# Predicting a new result with Polynomial regression
y_pred = predict(poly_reg, data.frame(Level = 6.5,
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3,
                                      Level4 = 6.5^4))