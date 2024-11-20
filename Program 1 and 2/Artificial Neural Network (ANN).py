# %% [markdown]
# Artificial Neural Network(ANN)

# %% [markdown]
# Importing Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# %%
tf.__version__


# %% [markdown]
# Part 1 - Data Preprocessing

# %% [markdown]
# Importing the Data set

# %%
df=pd.read_csv("Churn_Modelling.csv")
df.head()

# %%
x=df.iloc[:,3:-1].values
x

# %%
y=df.iloc[:,-1].values
y

# %% [markdown]
# Encoding The Categorical Data

# %% [markdown]
# Label Encoding The "Gender" Column

# %%
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])

# %%
x

# %% [markdown]
# One Hot Encoding The "Geography" Column

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
print(x)


# %% [markdown]
# Splitting The Data set into Training and Testing Sets

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)

# %%
x_train

# %%
x_test

# %%
y_test

# %%
y_train

# %% [markdown]
# Feature Scaling

# %%
#? we need to apply feature scaling to each and every variable in neural network
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_test=sc.fit_transform(x_test)
x_train=sc.transform(x_train)


# %%
x_train

# %%
x_test

# %% [markdown]
# Part 2 - Building the ANN (Artificial Neural Network)

# %% [markdown]
# Initializing the ANN

# %%
ann=tf.keras.models.Sequential()

# %% [markdown]
# Adding the Input layer and the first hidden layer

# %%
# shalow layer 
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

# %% [markdown]
# Adding the second hidden layer

# %%
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

# %% [markdown]
# Adding the Output layer

# %%
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
# ! for binary sigmoid
# ? for non binary softmax

# %% [markdown]
# Part 3 - Training The ANN

# %% [markdown]
# Compiling the ANN

# %%
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# ! binary outcome loss = " binary_crossentropy"
# ? for non binary "categorical_crossentropy"

# %% [markdown]
# Training The ANN model on the Training Set

# %%
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)
# ann.fit(x_train,y_train,batch_size=32,epochs=100)
# batch size is hyperparameter and given most used value 32



# %% [markdown]
# Part 4 - Making The Prediction and evaluating the model

# %% [markdown]
# Predicting result of a single observation

# %%
print(ann.predict(sc.transform([[1, 0, 0,600,1,40,3,60000,2,2,2,50000]])))
print(ann.predict(sc.transform([[1, 0, 0,600,1,40,3,60000,2,2,2,50000]]))>0.5)


# %% [markdown]
# Prdicting the results for x_test dataset

# %%
y_pred=ann.predict(x_test)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
y_pred=(y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# %% [markdown]
# Making The Confussion Matrix

# %%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sns
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='g')
plt.title("Confusion Matrix")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("confusion_matrix1.png")
plt.show()

# %% [markdown]
# Accuracy Score

# %%
print(accuracy_score(y_test,y_pred))

# %%
print(classification_report(y_test,y_pred))


