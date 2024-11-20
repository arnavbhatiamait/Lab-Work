# %% [markdown]
# Natural Language Processing (NLP) using different Classification Models

# %% [markdown]
# Importing the libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import re
import nltk
import seaborn as sns

# %% [markdown]
# Importing the data set

# %%
df=pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting=3)
df.head()

# %% [markdown]
# Cleaning the data set

# %%
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(1000):
    review=re.sub('[^a-zA-Z]',' ',df["Review"][i])
    review=review.lower()
    review=review.split()
    all_stopwords=stopwords.words("english")
    all_stopwords.remove('not')
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if word not in set(all_stopwords)]
    review=' '.join(review)
    corpus.append(review)
print(corpus)


# %% [markdown]
# Creating Bag of words Model

# %%
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=df.iloc[:,-1].values
print(x)
y


# %% [markdown]
# Splitting the dataset into training and testing data set

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0, test_size=0.2)

# %%
x_test

# %%
x_train

# %%
y_train

# %%
y_test

# %% [markdown]
# Logistic Regression 

# %%
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression()
classifier_LR.fit(x_train,y_train)
y_pred_LR = classifier_LR.predict(x_test)
print ("Logistic regression predicted and tested values \n" , np.concatenate((y_pred_LR.reshape(len(y_pred_LR),1),y_test.reshape(len(y_test),1)),1) )
print("Confussion Matrix of Logistic Regression")
cm_LR=confusion_matrix(y_test,y_pred_LR)
sns.heatmap(cm_LR, annot=True, fmt='g')
plt.title("Confussion Matrix of Logistic Regression")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("Confussion Matrix of Logistic Regression.png")
plt.show()
print("Accuracy Score: ", accuracy_score(y_pred_LR,y_test))
print("classification report :\n ", classification_report(y_pred_LR,y_test))


# %% [markdown]
# Naive Byes 

# %%
from sklearn.naive_bayes import GaussianNB
classifierNB=GaussianNB()
classifierNB.fit(x_train,y_train)
y_predNB=classifierNB.predict(x_test)
print(np.concatenate((y_predNB.reshape(len(y_predNB),1),y_test.reshape(len(y_test),1)),1))
cm_NB=confusion_matrix(y_test,y_predNB)
sns.heatmap(cm_NB, annot=True, fmt='g')
plt.title("Confusion Matrix  Naive Bayes")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("confusion matrix Naive Bayes.png")
plt.show()
print("accuracy score: \n", accuracy_score(y_test,y_predNB))
print("classification report :\n ",classification_report(y_test,y_predNB))

# %% [markdown]
# K Nearest Neighbors

# %%
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors=5)
classifier_KNN.fit(x_train,y_train)
y_pred_KNN = classifier_KNN.predict(x_test)
print ("K nearest Neighbours predicted and tested values \n" , np.concatenate((y_pred_KNN.reshape(len(y_pred_KNN),1),y_test.reshape(len(y_test),1)),1) )
print("Confussion Matrix of K nearest Neighbours")
cm_KNN=confusion_matrix(y_test,y_pred_KNN)
sns.heatmap(cm_KNN, annot=True, fmt='g')
plt.title("Confussion Matrix of K nearest Neighbours")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("Confussion Matrix of K nearest Neighbours.png")
plt.show()
print("Accuracy Score: ", accuracy_score(y_pred_KNN,y_test))
print("classification report :\n ", classification_report(y_pred_KNN,y_test))


# %% [markdown]
# Support Vector Machine

# %%
from sklearn.svm import SVC
classifier_SVM = SVC(kernel="linear",random_state=0)
classifier_SVM.fit(x_train,y_train)
y_pred_SVM = classifier_SVM.predict(x_test)
print ("Support Vector Machine predicted and tested values \n" , np.concatenate((y_pred_SVM.reshape(len(y_pred_SVM),1),y_test.reshape(len(y_test),1)),1) )
print("Confussion Matrix of Support Vector Machine")
cm_SVM=confusion_matrix(y_test,y_pred_SVM)
sns.heatmap(cm_SVM, annot=True, fmt='g')
plt.title("Confussion Matrix of Support Vector Machine")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("Confussion Matrix of Support Vector Machine.png")
plt.show()
print("Accuracy Score: ", accuracy_score(y_pred_SVM,y_test))
print("classification report :\n ", classification_report(y_pred_SVM,y_test))


# %% [markdown]
# Decission Tree

# %%
from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier_DT.fit(x_train,y_train)
y_pred_DT = classifier_DT.predict(x_test)
print ("Decission Trees predicted and tested values \n" , np.concatenate((y_pred_DT.reshape(len(y_pred_DT),1),y_test.reshape(len(y_test),1)),1) )
print("Confussion Matrix of Decission Trees")
cm_DT=confusion_matrix(y_test,y_pred_DT)
sns.heatmap(cm_DT, annot=True, fmt='g')
plt.title("Confussion Matrix of Decission Trees")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("Confussion Matrix of Decission Trees.png")
plt.show()
print("Accuracy Score: ", accuracy_score(y_pred_DT,y_test))
print("classification report :\n ", classification_report(y_pred_DT,y_test))


# %% [markdown]
# Random Forest Classification

# %%
from sklearn.ensemble import RandomForestClassifier
classifier_RFC = RandomForestClassifier(criterion="entropy",n_estimators=5,random_state=0)
classifier_RFC.fit(x_train,y_train)
y_pred_RFC = classifier_RFC.predict(x_test)
print ("Random Forest Classification predicted and tested values \n" , np.concatenate((y_pred_RFC.reshape(len(y_pred_RFC),1),y_test.reshape(len(y_test),1)),1) )
print("Confussion Matrix of Random Forest Classification")
cm_RFC=confusion_matrix(y_test,y_pred_RFC)
sns.heatmap(cm_RFC, annot=True, fmt='g')
plt.title("Confussion Matrix of Random Forest Classification")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("Confussion Matrix of Random Forest Classification.png")
plt.show()
print("Accuracy Score: ", accuracy_score(y_pred_RFC,y_test))
print("classification report :\n ", classification_report(y_pred_RFC,y_test))



