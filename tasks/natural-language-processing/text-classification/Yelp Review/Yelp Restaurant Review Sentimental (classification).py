
# coding: utf-8

# **Import library for ML project**

# In[46]:




# Machine learning library and packages

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.grid_search import GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')


# **Read Data**

# In[48]:


# Read yelp.csv data
df = pd.read_csv('yelp.csv')

#column name in data
print("Column are: ", df.columns)

# Shape of dataset
print("Shape of dataset: ", df.shape) # (10000, 10). 10k rows and 10 columns

# Initial few row of data
df.head()


# In[49]:


# From my own understandng they hide business Id, review_id, user_id for privacy reason. And this three features...
#..are not important for us


# **Basic EDA(Exploratory data analysis)**

# In[50]:


# Checking statistical details like percentile, mean, std etc. of a data frame df (that contains yelp.csv)
print("",df.describe())

# Summary of dataframe
print("Summary of df: ", df.info())


# In[51]:


# Checking length of each review. For that create a new column called "Review Len"
df['Review Len'] = df['text'].apply(len)

# Print initial dataset and we will "Reviw Len" column added as well. Print initial 3 row
df.head(3)


# In[71]:


# Class distribution of rating (in stars)
class_count = df.groupby('stars').size()
print(class_count)


# In[73]:


# basic statistics - mean, Standard deviation
# mean of rating (stars)
print("Mean is: ", df['stars'].mean())

# standard deviation. 
print("Standard Deviation: ", df['stars'].std())

"""According to Six sigma rule also known as the empirical rule or 68-95-99.7 rule. 
68% of your observations will fall between one standard deviation of the mean. 
95% will fall within two, and 99.7% will fall within three."""


# **Visualization**

# In[53]:


# Box plot 
sns.boxplot(x='stars', y='Review Len', data=df)


# In[54]:


#Visualization will help to know how features values are correlated to each other.

# Relation between review length and star rating. Histogram will be helpful.

Graph = sns.FacetGrid(data=df, col='stars')
Graph.map(plt.hist, 'Review Len', bins=40) # Text length is somewhat similar across all stars.


# In[55]:


# Now We will find if there's any correlation between useful, funny, and cool. Heatmap are useful for this task.
stars = df.groupby('stars').mean()
stars.corr()

sns.heatmap(data=stars.corr(), annot=True) 

# From heatmap. We can say. There is negative correlation between (cool and useful), (cool and funny), (cool and Review len)
# Positive realtion. (Funny and useful), (Review len & Funny), (Review len & Useful)


# In[56]:


# Total count of stars in dataset 
df['stars'].unique()
df['stars'].hist()

# Total number of count of rating 1, 2, 3, 4 and 5 

df['stars'].value_counts() # Lowest number total count is for  rating 1.


# In[57]:


# Select only rating with 1, 3, and 5.

df_class = df[(df['stars']==1) | (df['stars']==3) | (df['stars']==5)]
# df_class = df # accuracy was low with whole dataset
df_class.head()
print("Shape of dataset: ", df_class.shape) # only include row with rating 1, 3, and 5.

print(df_class)


# In[58]:


# seperate dataset.  X is input and y is output.
x = df_class['text']
y = df_class['stars']
# print(x.head())
# print(y.head())


# **Preprocessing of data**

# In[59]:


# library to import stop-word in English
from sklearn.feature_extraction import stop_words
 
# print(stop_words.ENGLISH_STOP_WORDS)


# In[60]:


# remove punctuation and stop-word.
import string
def text_preprocess(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    # remove stop-words from text
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[61]:


# Vectorization of text
"""convert review into vector form"""

bow_transformer = CountVectorizer(analyzer=text_preprocess).fit(x)
# len(bow_transformer.vocabulary_)


# In[62]:


x = bow_transformer.transform(x)

print('Sparse Matrix Shape: ', x.shape)
print('Amount of Non-Zero occurrences: ', x.nnz)


# **Training and test. Split dataset into two part- training and testing**

# In[63]:


# skleran for splitting dataset
from sklearn.model_selection import train_test_split

# sklearn function to split dataset into train, and test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)


# **Check performance of various Machine learning algorithm**

# In[64]:


# Logistic Regression

# import necessary library
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn import linear_model
from sklearn.metrics import recall_score

# mlflow library
import mlflow
import mlflow.sklearn

# Fitting Logistic Regression to the Training set
classifier = linear_model.LogisticRegression(C=1.5)
classifier.fit(x_train, y_train)

# test data - prediction
y_pred = classifier.predict(x_test)

# integrate MLflow
# with mlflow.start_run():

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix:\n",cm)

# Calculating Accuracy, Precision 
score_lr = accuracy_score(y_test,y_pred)
print("\n")
print("Accuracy is ", round(score_lr*100, 2),"%")
print("Classification report: ")
print(classification_report(y_test, y_pred))

# calculate recall and precision
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')

print("Precision: ", precision) # precision = true positive / true positive + false positive
print("recall: ", recall) # recall = True positive / True positive + False Negative

# Integrate ML Flow
mlflow.sklearn.log_model(classifier, "Model")
mlflow.log_metric("Accuracy", score_lr)
# In[65]:


# Using MNB(Multinomial Naive Bayes)
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
predmnb = mnb.predict(x_test)
print("Confusion Matrix for MNB:")
print(confusion_matrix(y_test,predmnb))
print("Score:", round(accuracy_score(y_test,predmnb)*100,2))
print("C", classification_report(y_test, predmnb))


# calculate recall and precision
precision = precision_score(y_test, predmnb, average='micro')
recall = recall_score(y_test, predmnb, average='micro')

print("Precision: ", precision) # precision = true positive / true positive + false positive
print("recall: ", recall) # recall = True positive / True positive + False Negative


# In[66]:


# Knn. K nearest neighbour

# import library for accuracy, recall and precision
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# K Nearest Neighbour Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)

# prediction on test data
pr_knn = knn.predict(x_test)
print("K neighbor confusion matrix:")
print(confusion_matrix(y_test, pr_knn))
print("Score: ", round(accuracy_score(y_test, pr_knn)*100,2))
print("Classification:")
print(classification_report(y_test, pr_knn))

# accuracy
score = accuracy_score(y_test, pr_knn)
print("\n")
print("Accuracy is ", round(score*100, 2),"%")

# calculate recall and precision
precision = precision_score(y_test, pr_knn, average='micro')
recall = recall_score(y_test, pr_knn, average='micro')

print("Precision: ", precision) # precision = true positive / true positive + false positive
print("recall: ", recall) # recall = True positive / True positive + False Negative


# In[67]:


# # MULTILAYER PERCEPTRON CLASSIFIER
# import MLP library from sklearn
from sklearn.neural_network import MLPClassifier
# import library for f1-score
from sklearn.metrics import f1_score

mlp = MLPClassifier()
mlp.fit(x_train, y_train)

# predict on test data
pr_mlp = mlp.predict(x_test)
print("Confusion Matrix:")
# create confusion matrix which is true positive, true negative, false positive, false negative 
print(confusion_matrix(y_test, pr_mlp))
score = round(accuracy_score(y_test,pr_mlp)*100,2)
print("Score is: ", score)
print("Classification Report:")
print(classification_report(y_test, pr_mlp))

# calculate recall and precision
precision = precision_score(y_test, pr_mlp, average='micro')
recall = recall_score(y_test, pr_mlp, average='micro')

print("Precision: ", precision) # precision = true positive / true positive + false positive
print("recall: ", recall) # recall = True positive / True positive + False Negative


# **Check ML model**

# ##Performance of Machine learning algorithm
# ###1) Logistic Regression:
#     Accuracy - 78.08
#     Precision - 0.78
# ###2) Multinomial Naive Bayes (MNB)
#     Accuracy - 74.23
#     Precision - 0.74
# ###3) K nearest Neighbour Algorithn
#     Accuracy - 60.66
#     Precision - 0.60
# ###4) Multilayer Perceptron Classifier
#     Accuracy - 78.38
#     Precision - 0.78

# In[68]:


# POSITIVE REVIEW
pr = df['text'][0]
print(pr[:100])
print("Actual Rating: ", df['stars'][0])
pr_t = bow_transformer.transform([pr])
print("Predicted Rating: ", mlp.predict(pr_t)[0])


# NOTE: MLflow integrated in python script of this notebook
