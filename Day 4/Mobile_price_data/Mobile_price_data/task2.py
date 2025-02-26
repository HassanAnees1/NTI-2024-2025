# Important Libraries
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import r2_score,f1_score,recall_score,precision_score,accuracy_score

# Data path 
path = r'C:\Users\moham\Desktop\NTI_ETA_CU\Session4\mobile_price_train.csv'
data = pd.read_csv(path)

print(data) # Show data 
data.corr() # correletion

# Features (4 only)
x = data[['battery_power', 'px_height' , 'px_width' , 'ram']] # features seletion 
y = data['price_range'] # target 

x_train , x_test , y_train , y_test = train_test_split( x , y , test_size=0.2 , random_state=42) # split data 

# Model LogisticRegression (keras)
model = Sequential()
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

results = model.fit(
    x_train,
    y_train,
    shuffle=True,  # Shuffle to not save the data not make overfitting 
    epochs=10,
    batch_size=16,
    validation_data=(x_test, y_test)  # Validation data to evaluate model performance
)
print(results.history) 
print('-'*20)

# Features (all)
x = data.drop('price_range' , axis='columns') # features seletion 
y = data['price_range'] # target 

# Model LogisticRegression (Keras)
model = Sequential()
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

results = model.fit(
    x_train,
    y_train,
    shuffle=True,  # Shuffle to not save the data not make overfitting 
    epochs=30,
    batch_size=16,
    validation_data=(x_test, y_test)  # Validation data to evaluate model performance
)

print(results.history)
print('-'*20)

# Features (4 only)
x = data[['battery_power', 'px_height' , 'px_width' , 'ram']] # features seletion 
y = data['price_range'] # target 

# Model LogisticRegression (sklearn)
model = LogisticRegression()
model.fit(x_train , y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test , y_pred)
f1 = f1_score(y_test , y_pred , average='micro')
recall = recall_score(y_test , y_pred , average='micro')
precision = precision_score(y_test , y_pred , average='micro')
accuracy = accuracy_score(y_test , y_pred)
 
print(f'R2 Accuracy : {r2*100:.2f} %')
print(f'F1 Accuracy : {f1*100:.2f} %')
print(f'Recall Accuracy : {recall*100:.2f} %')
print(f'Precision Accuracy : {precision*100:.2f} %')
print('-'*20)

# Features (all)
x = data.drop('price_range' , axis='columns') # features seletion 
y = data['price_range'] # target 

# model LogisticRegression (sklearn)
model = LogisticRegression()
model.fit(x_train , y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test , y_pred)
f1 = f1_score(y_test , y_pred , average='micro')
recall = recall_score(y_test , y_pred , average='micro')
precision = precision_score(y_test , y_pred , average='micro')
accuracy = accuracy_score(y_test , y_pred)
 
print(f'R2 Accuracy : {r2*100:.2f} %')
print(f'F1 Accuracy : {f1*100:.2f} %')
print(f'Recall Accuracy : {recall*100:.2f} %')
print(f'Precision Accuracy : {precision*100:.2f} %')
print('-'*20)
