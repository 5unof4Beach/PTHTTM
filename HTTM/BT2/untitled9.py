# import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
  
# importing data
df = pd.read_csv('headbrain1.csv')
print(df.shape)
  
# head of the data
print(df.head())
  
X= df['Head Size(cm^3)']
y=df['Brain Weight(grams)']
  
# using the train test split function
X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=104,train_size=0.8, shuffle=True)
  
# printing out train and test sets
print('X_train : ')
print(X_train.head())
print(X_train.shape)
print('')
print('X_test : ')
print(X_test.head())
print(X_test.shape)
print('')
print('y_train : ')
print(y_train.head())
print(y_train.shape)
print('')
print('y_test : ')
print(y_test.head())
print(y_test.shape)
