#%% Natural Language processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) # quoting = 3 is to ignore the double quotes ""

#%% Cleaning the texts
import re
import nltk
nltk.download("stopwords") # List of irrelevant words to remove

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # To get the roots of words

corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset["Review"][i]) # Remove symbols
    review = review.lower()
    review = review.split()
    ps = PorterStemmer() # Keep the root of the word
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # Remove irrelevant words
    review = ' '.join(review)
    corpus.append(review)


#%% Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1500) # we remove the unfrequented words in the corpus
x = cv.fit_transform(corpus).toarray()
y = dataset["Liked"].values

#%% Splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
# random_state is the seed
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# %% Fitting the Naive Bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# %% Predicting the test set results
y_pred = classifier.predict(x_test)
 
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)