# %% [markdown]
# Artificial Neural Network (ANN) Regression

# %% [markdown]
# Importing the libraries

# %%
pip install openpyxl

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# %%
tf.__version__ 

# %% [markdown]
# Part 1 Data Preprocessing

# %% [markdown]
# Importing the Dataset

# %%
df=pd.read_excel("Folds5x2_pp.xlsx")
df.head()

# %%
x=df.iloc[:,:-1].values
x

# %%
y=df.iloc[:,-1].values
y

# %% [markdown]
# Splitting The Data set into Training and Testing Set

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# %% [markdown]
# Part 2 Building the ANN

# %% [markdown]
# Initiating the ANN

# %%
ann=tf.keras.models.Sequential()

# %% [markdown]
# Adding the first Input layer and first Hidden layer

# %%
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

# %% [markdown]
# Adding the second Hidden layer

# %%
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

# %% [markdown]
# Adding the output layer

# %%
ann.add(tf.keras.layers.Dense(units=1))

# %% [markdown]
# Part 3 Training The ANN model

# %% [markdown]
# Compiling the ANN model

# %%
ann.compile(optimizer='adam',loss="mean_squared_error")

# %% [markdown]
# Training The ANN Model on the training Set

# %%
# ann.fit(x_train,y_train, batch_size=32,epchos=32)
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)


# %% [markdown]
# Predicting the results of the test set

# %%
y_pred=ann.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))



