# Multiple Linear regression

# Import data set
dataset = read.csv("50_Startups.csv")


# Encoding categorical data
dataset$State = factor(dataset$State, 
 levels=c('New York', 'California', 'Florida'),
 labels = c(1,2,3)
)

# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting multiple linear regression to the training set
# regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State)
regressor = lm(formula = Profit ~ ., data = training_set) # Dot means all the independent variables

print(summary(regressor))

# Predicting the Tests results
y_pred = predict(regressor, newdata = test_set)

# Building the optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration, 
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend, 
               data = dataset)
summary(regressor)
