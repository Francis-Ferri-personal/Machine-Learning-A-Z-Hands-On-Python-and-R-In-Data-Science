# Eclat

# Data preprocessing
# install.packages('arules')
library(arules)
dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Eclat on the dataset

# We need to set the minlen to two to not getting 1 item elements
rules = eclat(dataset, parameter = list(support = 0.0037, minlen=2))

# Visualizing the results
# This does not obtain rules, It obtains sets
inspect(sort(rules, by = 'support')[1:10])
