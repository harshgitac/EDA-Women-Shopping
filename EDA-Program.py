#!/usr/bin/env python
# coding: utf-8

# # This dataset includes 23486 rows and 10 feature variables. Each row corresponds to a customer review, and includes the variables:
# 
# Clothing ID: Integer Categorical variable that refers to the specific piece being reviewed.
# Age: Positive Integer variable of the reviewers age.
# Title: String variable for the title of the review.
# Review Text: String variable for the review body.
# Rating: Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
# Recommended IND: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.
# Positive Feedback Count: Positive Integer documenting the number of other customers who found this review positive.
# Division Name: Categorical name of the product high level division.
# Department Name: Categorical name of the product department name.
# Class Name: Categorical name of the product class name.
# 

# In[1]:


get_ipython().system('pip install textblob')
#used for POs tagging aswell as sentiment analysis


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import plotly as py #  open-source plotting library that supports over 40 unique chart types covering a wide range of statistical, financial, geographic, scientific, and 3-dimensional use-cases.
import cufflinks as cf #Cufflinks connect Plotly with pandas to create graphs and charts of Dataframes directly. Its a Python library which is used to design graphs, especially interactive graphs. 


# In[4]:


from plotly.offline import iplot #For offline use


# In[5]:


py.offline.init_notebook_mode(connected=True)
cf.go_offline()


# Importing data

# In[6]:


df=pd.read_csv("F:\Womens Clothing E-Commerce Reviews.csv",index_col=0)
df.head()


# In[7]:


df.drop(labels=['Title','Clothing ID'],axis =1, inplace =True)
#Dropped unused Title, Clothing ID labels


# In[8]:


df.head(20)


# In[9]:


df.isnull().sum() # To check null values number


# In[10]:


df.dropna(subset=['Review Text','Division Name'],inplace= True)


# In[ ]:





# In[11]:


df.isnull().sum()


# In[12]:


' '.join(df['Review Text'].tolist()) #Review to join in a single list


# # Text Cleaning

# In[13]:


contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": " he will",
"he'll've": " he will have",
"he's": "he has",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": " how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}


# In[14]:


# Method to change contractions to expressions
def con_to_exp(x):
    if type(x) is str:
        x=x.replace('\\','')
        for key in contractions:
            value = contractions[key]
            x= x.replace(key,value)
        return x
    else:
        return x


# In[15]:


get_ipython().run_cell_magic('time', '', "# magic command to estimate time\n# Using lambda method\ndf['Review Text']=df['Review Text'].apply(lambda x:con_to_exp(x))")


# In[16]:


' '.join(df['Review Text'].tolist()) #Recheck the text


# # Feature Engineering

# In[17]:


#Sentiment polity- defines the orientation of the expressed sentiment, i.e., it determines if the text expresses the positive, negative or neutral sentiment of the user about the entity in consideration.
from textblob import TextBlob


# In[18]:


df.head()


# In[19]:


df['polarity']=df['Review Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
#TextBlob is a Python (2 and 3) library for processing textual data. It provides a simple API for diving into common natural language processing (NLP) tasks 


# In[20]:


df['review_len']=df['Review Text'].apply(lambda x: len(x)) 
#To calculate total number of characters in review


# In[21]:


df['word_count']=df['Review Text'].apply(lambda x: len(x.split()))
#To calculate total number of words in each review


# In[22]:


#To calculate average word count in Review
def get_avg_word_len(x):
    words =x.split()
    word_len=0
    for word in words:
        word_len=word_len+len(word)
        
    return word_len/len(words)


# In[23]:


df['avg_word_len']= df['Review Text'].apply(lambda x: get_avg_word_len(x))


# In[24]:


df.head()


#    # Distribution of Sentiment Polarity

# In[25]:


df['polarity'].iplot(kind ='hist',color ='orange',bins=50,xTitle='Polarity',yTitle='Count',title="Sentiment Polarity Distribution")


# # Distribution of Reviews Rating and Reviewers Age

# In[26]:


#
df['Rating'].iplot(kind='hist',color='Blue',linecolor='yellow',xTitle='Rating',yTitle='Count',title='Review rating Distribution')


# In[27]:


df['Age'].iplot(kind='hist',bins=40,colors='aqua',linecolor='yellow',xTitle='Age',yTitle='Count',title='Reviewers Age Distribution')


# 
# #  Distribution of Review Text Length  and Word Length

# In[28]:


df['review_len'].iplot(kind='hist',xTitle='Review Len',yTitle='Count',title='Review text Length')


# In[29]:


df['word_count'].iplot(kind='hist',color='darkcyan',xTitle='Word Count',yTitle='Count',title='Word Count Distribution')


# In[30]:


df['avg_word_len'].iplot(kind='hist',color='darkturquoise',linecolor='Red',xTitle='Average Word Len',yTitle='Count',title='Review Text Avg Word Length Distribution')


# # Distribution of Categorial Values- Department,Division, Class
# 

# In[31]:


df['Department Name'].value_counts().iplot(kind='bar',color='grey',xTitle='Department',yTitle='Count', title="Bar Chart of Department's Name")


# In[32]:


df['Division Name'].value_counts().iplot(kind='bar',color='black',linecolor='red',xTitle='Division',yTitle='Count', title="Bar Chart of Division's Name")


# In[33]:


df['Class Name'].value_counts().iplot(kind='bar',color='Darkblue',linecolor='blue',xTitle='Class',yTitle='Count', title="Bar Chart of Class's Name")


# # Distribution of Unigram, Bigram and Trigram with Stop-Words

# In[34]:


from sklearn.feature_extraction.text import CountVectorizer
#Convert a collection of text documents to a matrix of token counts. This implementation produces a sparse representation of the counts using scipy.sparse.


# In[35]:


def get_top_n_words(x,n):
    vec = CountVectorizer().fit(x)
    bow = vec.transform(x)
    sum_words=bow.sum(axis=0)
    words_freq=[(word, sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
    words_freq=sorted(words_freq,key=lambda x:x[1],reverse =True)
    return words_freq[:n]


# In[36]:


words= get_top_n_words(df['Review Text'],20)


# In[37]:


words


# In[38]:


df1=pd.DataFrame(words,columns=['Unigram','Frequency'])
df1=df1.set_index('Unigram')
df1.iplot(kind='bar',xTitle='Unigram',yTitle='Count',color='Red',title='Top 20 Unigram words in Reviews')


# 
# # Bigram

# In[39]:


def get_top_n_words(x,n):
    vec = CountVectorizer(ngram_range=(2,2)).fit(x)
    bow = vec.transform(x)
    sum_words=bow.sum(axis=0)
    words_freq=[(word, sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
    words_freq=sorted(words_freq,key=lambda x:x[1],reverse =True)
    return words_freq[:n]


# In[40]:


words= get_top_n_words(df['Review Text'],20)


# In[41]:


words


# In[42]:


df1=pd.DataFrame(words,columns=['Bigram','Frequency'])
df1=df1.set_index('Bigram')
df1.iplot(kind='bar',xTitle='Bigram',yTitle='Count',color='Grey',linecolor='blue',title='Top 20 Bigram words in Reviews')


# 
# # Trigram

# In[43]:




def get_top_n_words(x,n):
    vec = CountVectorizer(ngram_range=(3,3)).fit(x)
    bow = vec.transform(x)
    sum_words=bow.sum(axis=0)
    words_freq=[(word, sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
    words_freq=sorted(words_freq,key=lambda x:x[1],reverse =True)
    return words_freq[:n]


words= get_top_n_words(df['Review Text'],20)

words

df1=pd.DataFrame(words,columns=['Trigram','Frequency'])
df1=df1.set_index('Trigram')
df1.iplot(kind='bar',xTitle='Trigram',yTitle='Count',color='Blue',linecolor='black',title='Top 20 Trigram words in Reviews')


# # Distribution of Unigram, Bigram and Trigram without Stop-Words

# ## Unigram

# In[44]:




def get_top_n_words(x,n):
    vec = CountVectorizer(ngram_range=(1,1),stop_words='english').fit(x)
    bow = vec.transform(x)
    sum_words=bow.sum(axis=0)
    words_freq=[(word, sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
    words_freq=sorted(words_freq,key=lambda x:x[1],reverse =True)
    return words_freq[:n]


words= get_top_n_words(df['Review Text'],20)

words

df1=pd.DataFrame(words,columns=['Unigram','Frequency'])
df1=df1.set_index('Unigram')
df1.iplot(kind='bar',xTitle='Unigram',yTitle='Count',color='Blue',linecolor='black',title='Top 20 Unigram(excluding Stop words) words in Reviews')


# ## Bigram

# In[45]:



def get_top_n_words(x,n):
    vec = CountVectorizer(ngram_range=(2,2),stop_words='english').fit(x)
    bow = vec.transform(x)
    sum_words=bow.sum(axis=0)
    words_freq=[(word, sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
    words_freq=sorted(words_freq,key=lambda x:x[1],reverse =True)
    return words_freq[:n]


words= get_top_n_words(df['Review Text'],20)

words

df1=pd.DataFrame(words,columns=['Bigram','Frequency'])
df1=df1.set_index('Bigram')
df1.iplot(kind='bar',xTitle='Bigram',yTitle='Count',color='black',linecolor='red',title='Top 20 Bigram(excluding Stop words) words in Reviews')


# # Trigram

# In[46]:



def get_top_n_words(x,n):
    vec = CountVectorizer(ngram_range=(3,3),stop_words='english').fit(x)
    bow = vec.transform(x)
    sum_words=bow.sum(axis=0)
    words_freq=[(word, sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
    words_freq=sorted(words_freq,key=lambda x:x[1],reverse =True)
    return words_freq[:n]


words= get_top_n_words(df['Review Text'],20)

words

df1=pd.DataFrame(words,columns=['Trigram','Frequency'])
df1=df1.set_index('Trigram')
df1.iplot(kind='bar',xTitle='Trigram',yTitle='Count',color='yellow',linecolor='red',title='Top 20 Trigram(excluding Stop words) words in Reviews')


# # Distribution of Top 20 POS tags

# In[47]:


get_ipython().system('pip install nltk')


# In[48]:


import nltk
nltk.download('punkt')
#pacakge of nltk This tokenizer divides a text into a list of sentences by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences. 
nltk.download('averaged_perceptron_tagger')
#averaged_perceptron_tagger is used for tagging words with their parts of speech (POS).


# In[49]:


blob=TextBlob(str(df['Review Text']))


# In[50]:


nltk.download('tagsets')
print(nltk.help.upenn_tagset())
#tags 


# In[51]:


pos_df= pd.DataFrame(blob.tags,columns=['words','pos'])
pos_df= pos_df['pos'].value_counts()
#Full form can be shown in nltk.help.upenn_tagset()
pos_df.iplot(kind='bar',color='black',linecolor='red',xTitle='Part of Speech', yTitle='Count', title='Distribution of Part of Speech Tags')


#  # Bivariate Analysis

# In[52]:


sns.pairplot(df)
#To plot all the numerical columns in form of pair plot 


# In[53]:


# Categorigal Plot
sns.catplot(x='Division Name',y='polarity',data=df)


# In[54]:


# Box Plot
sns.catplot(x='Division Name',y='polarity',data=df,kind='box')


# In[55]:


# Categorigal Plot for Department
sns.catplot(x='Department Name',y='polarity',data=df)


# In[56]:


# Box Plot
sns.catplot(x='Department Name',y='polarity',data=df,kind='box')


# In[57]:


# Division name vs Review Lengh
# Box Plot
sns.catplot(x='Division Name',y='review_len',data=df,kind='box')


# In[58]:


# Department name vs Review Lengh
# Box Plot
sns.catplot(x='Department Name',y='review_len',data=df,kind='box')


# In[59]:


import plotly.express as px
import plotly.graph_objects as go


# In[60]:


x1=df[df['Recommended IND']==1]['polarity']

x0=df[df['Recommended IND']==0]['polarity']


# In[61]:


trace1= go.Histogram(x=x0,name='Not Recommended',opacity=0.9)
trace2= go.Histogram(x=x1,name='Recommended',opacity=0.7)


# In[62]:


data=[trace1,trace2]
layout=go.Layout(barmode='overlay',title='Distribution of Sentiment Polarity of Reviews Based on Recommendation')
fig=go.Figure(data=data,layout=layout)
iplot(fig)


# # Distribution of Ratings based on the Recommendation

# In[63]:


import plotly.express as px
import plotly.graph_objects as go


x1=df[df['Recommended IND']==1]['Rating']

x0=df[df['Recommended IND']==0]['Rating']


trace1= go.Histogram(x=x0,name='Not Recommended',opacity=0.9)
trace2= go.Histogram(x=x1,name='Recommended',opacity=0.7)

data=[trace1,trace2]
layout=go.Layout(barmode='overlay',title='Distribution of Reviews Rating Based on Recommendation')
fig=go.Figure(data=data,layout=layout)
iplot(fig)


# In[64]:


#Join Plot
sns.jointplot(x='polarity',y='review_len',data=df,kind='kde')


# In[65]:


#Join Plot- Polarity and Age
sns.jointplot(x='polarity',y='Age',data=df,kind='kde')


# In[66]:


df.iat[5,1]


# In[67]:


# To get particular matching value
d2 = df[(df['Rating']==5)


# In[ ]:




