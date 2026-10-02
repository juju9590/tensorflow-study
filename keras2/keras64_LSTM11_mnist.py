#36-3 copy

import numpy as np
import pandas as pd
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, LSTM
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, MaxPool2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint ,ReduceLROnPlateau

#1. 데이터
(x_train, y_train), (x_test, y_test)= mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test)) # 255 0

#### 스케일링 1
x_train = x_train/255. # .만 붙이면 float 형태로 출력하게 됨
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) # 1.0 0.0 (0~1 사이로 나옴)
print(np.max(x_test), np.min(x_test)) # 1.0 0.0
# MinMaxScaler, MaxAbsScaler 와 동일, 이미지기 때문에 255로 나누면 동일한 값이 나온다.

##### X reshape  하는 이유 : input_shape 
x_train = x_train.reshape(-1,28,28,1) 
x_test = x_test.reshape(-1,28,28,1)
print(x_train.shape ,x_test.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)

##### y값 알아보기

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))

### RNN 모델로 연결하기 위해 3차원으로 다시 reshape
x_train = x_train.reshape(-1,28,28) 
x_test = x_test.reshape(-1,28,28)
print(x_train.shape ,x_test.shape) #(60000, 28, 28) (10000, 28, 28)

#2. 모델구성
model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(28,28))) 

model.add(LSTM(64, return_sequences=True,))
model.add(LSTM(32))
model.add(Dense(64, activation='relu'))
model.add(Dense(32)) 
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))  

model.summary()

#3. 컴파일, 훈련
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',
              metrics = ['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    verbose=1,
    patience=20,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5, # learning_rate(러닝레이트) 비율 조절

)

import datetime
date = datetime.datetime.now()
date = date.strftime('%d%m-%H%M')

path ='./_save/keras36/'

filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k36_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
    filepath=filepath,
    mode='min',
)

start_time=time.time()
model.fit(x_train,y_train,
          epochs=1, 
          batch_size=128, 
          verbose=1,
          validation_split=0.2,
          callbacks = [es, mcp, rlr],
          )
end_time=time.time()

# 4. 평가, 예측
print( "=============model.evaluate=================")
loss = model.evaluate(x_test, y_test, verbose=1)

print('loss : ', round(loss[0],2))
print('acc : ', round(loss[1],2))

print(x_test.shape) #(10000, 28, 28)
print(y_test.shape) #(10000,)

y_pred = model.predict(x_test)
print(y_pred.shape)  #(10000, 10)

y_pred = np.argmax(y_pred, axis=1)
print(y_pred.shape) # (10000,)
print(y_test.shape) # (10000,)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)
print ('걸린시간 : ', round(end_time-start_time,2),'초' )

### 결과
# loss :  0.02
# acc :  0.99
# acc_score :  0.9928
# 걸린시간 :  79.84 초


















