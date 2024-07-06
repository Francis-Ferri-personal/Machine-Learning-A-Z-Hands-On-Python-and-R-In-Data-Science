#%% Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

#%% Training Apriori on the dataset
from apyori import apriori

# These are the items that are bought at least three times per day in a week.
# A very high minimal confidence probably is going to return only the most purchased items and probably there is not a real relation between them. Those items are just purchased a lot.
rules = apriori(transactions, min_support=3*7/len(transactions), min_confidence=0.2, min_lift=3, min_length = 2)

#%% Visualizing the results
# These rules are already sorted by the relevance. The relevance is a criteria confined by support, lift etc.
results = list(rules)