import numpy as np
import pandas as pd
df = pd.read_csv('D:/btBoxung/diemthi.csv')
df.info()

df = df.drop(labels='Họ Tên', axis=1)
#---check for null values---
print("Nulls")
print("=====")
print(df.isnull().sum())


#---check for 0s---
print("0s")
print("==")
print(df.eq(0).sum())

corr = df.corr()
print(corr)

X = df[['10%','20%','20%.1']]
y = df.iloc[:,3]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(12, input_dim=3, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"]) 

history=model.fit(X_train, y_train,epochs=10,batch_size=10,validation_split=0.3, verbose=0)
print(history.history.keys())
y_pred = model.predict(X_test)
print(y_pred.size)
# =============================================================================
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test, y_pred))
# =============================================================================


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()






