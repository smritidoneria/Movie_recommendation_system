#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[4]:


movies.head(1)


# In[9]:


credits.head()


# In[13]:


movies=movies.merge(credits,on='title')


# In[14]:


movies.head()


# In[17]:


# genres
# id
# keywords
# title
# overview
# cast 
# crew 
movies=movies [['id','keywords','title','overview','cast','crew','genres']]


# In[18]:


movies.head()


# In[22]:


# to check if we have any vacant column
movies.isnull().sum()


# In[21]:


# remove the movies that have nan values. parameter suggests that do the command on this database itself. do not create another.

movies.dropna(inplace=True)


# In[23]:


movies.duplicated().sum()


# In[25]:


# iloc function is used in pandas to access the row and column in the database using the integer index.
movies.iloc[0].genres


# In[35]:


# ast is the module in python which is used to evaluate python expressions from strings.
import ast
def convert(obj):
    L=[];
    for i in ast.literal_eval(obj):
            L.append(i['name'])
    return L    


# In[37]:


movies['genres']=movies['genres'].apply(convert)


# In[40]:


movies.head()


# In[41]:


movies['keywords']=movies['keywords'].apply(convert)


# In[42]:


movies.head()


# In[43]:


movies['cast'][0]


# In[47]:


import ast
def convert1(obj):
    L=[];
    count=0;
    for i in ast.literal_eval(obj):
        if(count!=3):
            L.append(i['name'])
            count+=1
        else:
            break
    return L 


# In[48]:


movies['cast']=movies['cast'].apply(convert1)


# In[49]:


movies['cast'][0]


# In[50]:


movies['crew'][0]


# In[54]:


import ast
def convert2(obj):
    L=[];
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break;
    return L 


# In[57]:


movies['crew']=movies['crew'].apply(convert2)


# In[58]:


movies.head()


# In[59]:


movies['overview']


# In[61]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[62]:


movies.head()


# In[64]:


movies['genres']=movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])


# In[110]:


movies.head()


# In[66]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[112]:


movies.head()


# In[113]:


new_df=movies[['id','title','tags']]


# In[114]:


new_df.head()


# In[72]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[111]:


new_df.head()


# In[74]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[75]:


new_df.head()


# In[83]:


# nltk library is used to get the rid from the repeated words. like love loved loving, these are all repeated right, so by using this library we can get the output as love only.
import nltk


# In[85]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[88]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)    


# In[90]:


new_df['tags']=new_df['tags'].apply(stem)


# In[91]:


# countvectorizer is used to count the most frequent words from the tags.  but it will do it first it will combine all the tags of the movies. 
# now it will form the table of words(5000) and shows which word is frequenly there. stop_words means it is a list containing the words like of , the, in etc. it will count those words.


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words="english")


# In[92]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[93]:


vectors


# In[94]:


vectors[0]


# In[95]:


cv.get_feature_names_out()


# In[96]:


ps.stem('loved')


# In[97]:


# now, what we are doing is we are using cosine_similarity to find the similarity between the vectors. cosine_similarity
# generally refers to the similarity index bween 0 and 1.
from sklearn.metrics.pairwise import cosine_similarity


# In[99]:


similarity=cosine_similarity(vectors)


# In[101]:


similarity.shape


# In[123]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[103]:


similarity[1]


# In[130]:


def recommend(movie):
    movie_index=new_df[new_df['title']=='Avatar'].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        


# In[118]:


new_df[new_df['title']=='Avatar'].index[0]


# In[131]:


recommend('Avatar')


# In[127]:


new_df.iloc[1216].title


# In[132]:


import pickle


# In[133]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[136]:


new_df.to_dict()


# In[137]:


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))


# In[138]:


pd.DataFrame(new_df.to_dict())


# In[139]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




