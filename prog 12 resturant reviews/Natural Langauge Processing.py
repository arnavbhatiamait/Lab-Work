# %% [markdown]
# Natural Langauge Processing (NLP)

# %% [markdown]
# Importing the libraries

# %%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# Importing The Dataset

# %%
dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting=3)
#!\t is used while working with tab seperated value
#? quoting is used to avoid the quotes in the file
dataset.head()

# %% [markdown]
# Cleaning The Data set

# %%
import re 
import nltk
# ! Natural language ToolKit library
nltk.download('stopwords')
from nltk.corpus import stopwords
# ? stopwords are articles like a an the to etc
from nltk.stem.porter import PorterStemmer
corpus=[]
# ! for loop to make each review seperately
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    # ! replacing all non letters with spaces
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    # ? removing not from the list of stopwords
    all_stopwords=stopwords.words('english')
    all_stopwords.remove('not')
    # * used to replace words to their original character
    # review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review=' '.join(review)
    corpus.append(review)
    print(review)
print(corpus) 

# %% [markdown]
# Creating Bag Of Words Model

# %%
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,-1].values
print(x)
y


# %% [markdown]
# Splitting the Dataset into Training And Test set

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# %%
x_test

# %%
x_train

# %%
y_test

# %%
y_train

# %% [markdown]
# Training the Naive Byes Model On the training Set

# %%
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

# %% [markdown]
# Predicting the test set results on the test set

# %%
y_predNB=classifier.predict(x_test)
print(np.concatenate((y_predNB.reshape(len(y_predNB),1),y_test.reshape(len(y_test),1)),1))

# %% [markdown]
# Confussion Matrix

# %%
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
import seaborn as sns
cm=confusion_matrix(y_test,y_predNB)
sns.heatmap(cm, annot=True, fmt='g')
plt.title("Confusion Matrix")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("confusion_matrix.png")
plt.show()

# %% [markdown]
# Accuracy Score

# %%
print(accuracy_score(y_test,y_predNB))

# %%
print(classification_report(y_test,y_predNB))


