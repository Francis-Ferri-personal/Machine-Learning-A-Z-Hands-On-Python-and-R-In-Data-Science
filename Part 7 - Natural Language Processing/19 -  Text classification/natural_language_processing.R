# Natural Language processing

# Importing the dataset
dataset_original = read.delim("Restaurant_Reviews.tsv", quote = '', stringsAsFactors = FALSE)

# Cleaning the text
# install.packages("tm")
# install.packages("SnowballC")
library(tm)
library(SnowballC)

corpus = VCorpus(VectorSource(dataset_original$Review))

corpus = tm_map(corpus, content_transformer(tolower)) # To lower case
corpus = tm_map(corpus, removeNumbers) # Remove numbers
corpus = tm_map(corpus, removePunctuation) # Remove punctuation
corpus = tm_map(corpus, removeWords, stopwords()) # Remove irrelevant words
corpus = tm_map(corpus, stemDocument)# Get the root of the words
corpus = tm_map(corpus, stripWhitespace)# Remove white spaces


# Creating the bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999) # Remove infrequent words

dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked


# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# install.packages('caTools')

# Spliting the datasetset into Training set and Test set
library(caTools)
set.seed(123)
# Split ratio is for training set in this case not for test
# The column used is y
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

# Fitting Random Forest Classification to the training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692], y = training_set$Liked, ntree=10)

# Predicting the Test set results
y_pred = predict(classifier, type='response', newdata = test_set[-692])

# Making the confusion matrix
cm = table(test_set[,692], y_pred)
