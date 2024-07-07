# %% Upper Confidence bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# This example uses click rate over advertisements. Each row is if the user clicked the advertisement
#%% Implement the UCB from scratch
N = 10000
d = dataset.shape[1] # 10

ad_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

for n in range(N):
    max_upper_bound = 0
    ad = 0
    for i in range(d):
        if(numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1)/ numbers_of_selections[i]) # +1 because the first value is zero
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    
    ad_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward


#%% Visualizing the results
plt.hist(ad_selected)
plt.title("Histogram of adselections")
plt.xlabel("Add")
plt.ylabel("Number of times aech ad was selected")
plt.show()