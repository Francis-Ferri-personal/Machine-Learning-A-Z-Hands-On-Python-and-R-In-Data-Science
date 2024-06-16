# Simple Linear regresion

# Import dataset
dataset = read.csv('Salary_Data.csv')
# Taking care of missing data\


# install.packages('caTools')

# Spliting the datasetset into Training set and Test set
library(caTools)
set.seed(123)
# Split ratio is for training set in this case not for test
# The column used is y
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)


# Feature scaling
# Not needed because the package does this for us
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])

regressor = lm(formula= Salary ~ YearsExperience, 
               data = training_set)

# Predict the test set results
y_pred = predict(regressor, newdata =  test_set)



# install.packages("ggplot2", dependencies=TRUE)
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red' ) +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), 
            colour = 'blue') +
  ggtitle('Salary vs Years of experience (Training)') +
  xlab('Years of experience') +
  ylab('Salary')

# Visualizing the test set
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red' ) +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), 
            colour = 'blue') +
  ggtitle('Salary vs Years of experience (Test)') +
  xlab('Years of experience') +
  ylab('Salary')
