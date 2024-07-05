# Apriori

# Data preprocessing
# install.packages('arules')
library(arules)
dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ',', rm.duplicates = TRUE)
summary(dataset, top)
itemFrequencyPlot(dataset, topN = 10)

# Training apriori on the dataset

# you have to set your minimum support and confidence based on your business problem and objective. it is subjective
rules = apriori(dataset, parameter = list(support = 0.0037, confidence =  0.2))

# Visualizing the results
inspect(sort(rules, by = 'lift')[1:10])
