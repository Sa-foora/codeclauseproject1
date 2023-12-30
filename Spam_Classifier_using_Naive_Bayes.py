#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PratikhyaManas/Spam-Classifier-using-Naive-Bayes/blob/master/Spam_Classifier_using_Naive_Bayes.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # PROBLEM STATEMENT

# - The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.
# 
# - The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.
# 

# # STEP #0: LIBRARIES IMPORT
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # STEP #1: IMPORT DATASET

# In[4]:


from google.colab import files
uploaded = files.upload()
import io
spam_df = pd.read_csv(io.BytesIO(uploaded['emails.csv']))


# In[5]:


spam_df.head(10)


# In[6]:


spam_df.tail()


# In[7]:


spam_df.describe()


# In[8]:


spam_df.info()


# # STEP #2: VISUALIZE DATASET

# In[9]:


# Let's see which message is the most popular ham/spam message
spam_df.groupby('spam').describe()


# In[10]:


# Let's get the length of the messages
spam_df['length'] = spam_df['text'].apply(len)
spam_df.head()


# In[11]:


spam_df


# In[12]:


spam_df['length'].plot(bins=100, kind='hist')


# In[13]:


spam_df.length.describe()


# In[14]:


# Let's see the longest message 43952
spam_df[spam_df['length'] == 43952]['text'].iloc[0]


# In[ ]:


# Let's divide the messages into spam and ham


# In[ ]:


ham = spam_df[spam_df['spam']==0]


# In[ ]:


spam = spam_df[spam_df['spam']==1]


# In[17]:


ham


# In[18]:


spam


# In[19]:


spam['length'].plot(bins=60, kind='hist') 


# In[20]:


ham['length'].plot(bins=60, kind='hist') 


# In[21]:


print( 'Spam percentage =', (len(spam) / len(spam_df) )*100,"%")


# In[22]:


print( 'Ham percentage =', (len(ham) / len(spam_df) )*100,"%")


# In[23]:


sns.countplot(spam_df['spam'], label = "Count") 


# # STEP #3: CREATE TESTING AND TRAINING DATASET/DATA CLEANING

# # STEP 3.1 REMOVE PUNCTUATION

# In[24]:


import string
string.punctuation


# In[ ]:


Test = 'Hello Mr. Future, I am so happy to be learning AI now!!'


# In[26]:


Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed


# In[28]:


# Join the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# # STEP 3.2 REMOVE STOPWORDS

# In[29]:


# Download stopwords Package to execute this command
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')


# In[30]:


Test_punc_removed_join


# In[ ]:


Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]


# In[32]:


Test_punc_removed_join_clean # Only important (no so common) words are left


# # STEP 3.3 COUNT VECTORIZER EXAMPLE 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)


# In[35]:


print(vectorizer.get_feature_names())


# In[36]:


print(X.toarray())  


# # APPLY THE PREVIOUS THREE PROCESSES TO OUR SPAM/HAM EXAMPLE

# In[ ]:


# Define a pipeline to clean up all the messages 
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


# In[ ]:


# Test the newly added function
spam_df_clean = spam_df['text'].apply(message_cleaning)


# In[39]:


print(spam_df_clean[0])


# In[40]:


print(spam_df['text'][0])


# # APPLY COUNT VECTORIZER TO OUR MESSAGES LIST

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning)
spamham_countvectorizer = vectorizer.fit_transform(spam_df['text'])


# In[42]:


print(vectorizer.get_feature_names())


# In[43]:


print(spamham_countvectorizer.toarray())  


# In[44]:


spamham_countvectorizer.shape


# # STEP#4: TRAINING THE MODEL WITH ALL DATASET

# In[45]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
label = spam_df['spam'].values
NB_classifier.fit(spamham_countvectorizer, label)


# In[ ]:


testing_sample = ['Free money!!!', "Hi Kim, Please let me know if you need any further information. Thanks"]
testing_sample_countvectorizer = vectorizer.transform(testing_sample)


# In[47]:


test_predict = NB_classifier.predict(testing_sample_countvectorizer)
test_predict


# In[ ]:


# Mini Challenge!
testing_sample = ['Hello, I am Ryan, I would like to book a hotel in Bali by January 24th', 'money viagara!!!!!']


# In[49]:


testing_sample = ['money viagara!!!!!', "Hello, I am Ryan, I would like to book a hotel in SF by January 24th"]
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
test_predict


# # STEP#4: DIVIDE THE DATA INTO TRAINING AND TESTING PRIOR TO TRAINING

# In[ ]:


X = spamham_countvectorizer
y = label


# In[51]:


X.shape


# In[52]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[63]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# In[ ]:


#from sklearn.naive_bayes import GaussianNB 
#NB_classifier = GaussianNB()
#NB_classifier.fit(X_train, y_train)


# # STEP#5: EVALUATING THE MODEL 

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[65]:


y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


# In[66]:


# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[67]:


print(classification_report(y_test, y_predict_test))


# # STEP #6: LET'S ADD ADDITIONAL FEATURE TF-IDF

# In[68]:


spamham_countvectorizer


# In[69]:


from sklearn.feature_extraction.text import TfidfTransformer

emails_tfidf = TfidfTransformer().fit_transform(spamham_countvectorizer)
print(emails_tfidf.shape)


# In[70]:


print(emails_tfidf[:,:])
# Sparse matrix with all the values of IF-IDF


# In[71]:


X = emails_tfidf
y = label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix
y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


# In[72]:


print(classification_report(y_test, y_predict_test))

