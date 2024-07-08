#%% Thompson sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# This example uses click rate over advertisements. Each row is if the user clicked the advertisement
#%% Implement the Thompson sampling from scratch
N = 10000
d = dataset.shape[1] # 10

ad_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d

total_reward = 0

for n in range(N):
    max_random = 0 # Maximum random draw
    ad = 0 
    for i in range(d):
        """
        At the beginning all starts equal
        In each round, the algorithm must decide which option to select, balancing between exploring lesser-known options (exploration) and exploiting the options that have proven to be good (exploitation).

        This means that the options with more successes in the past (higher values of numbers_of_rewards_1) tend to have higher random_beta values, but there is still a possibility of selecting lesser-explored options due to the inherent variability of the Beta distribution.
        """
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    
    ad_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    total_reward += reward


#%% Visualizing the results
plt.hist(ad_selected)
plt.title("Histogram of ad selections")
plt.xlabel("Add")
plt.ylabel("Number of times aech ad was selected")
plt.show()