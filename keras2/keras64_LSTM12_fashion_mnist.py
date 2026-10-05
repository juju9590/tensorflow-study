#53-12 카피
# 실습 : acc 0.92 목표

import numpy as np
import pandas as pd
import time

from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, LSTM, Reshape, GRU, SimpleRNN
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터 
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# print(np.max(x_train), np.min(x_train))
# print(np.max(x_test), np.min(x_test))

### 스케일링 (x_train 만)
x_train = x_train/255.0
x_test = x_test/255.0

### x-reshape (4차원) CNN 모델 적용 
# x_train = x_train.reshape(-1,28,28,1)
# x_test = x_test.reshape(-1,28,28,1)
# print(x_train.shape, x_test.shape) #(60000, 28, 28, 1) (10000, 28, 28, 1)

### x-reshape (2차원) DNN 모델 적용
x_train = x_train.reshape(-1,28*28*1)
x_test = x_test.reshape(-1,28*28*1)
# print(x_train.shape, x_test.shape) #(60000, 784) (10000, 784)

### x-reshape (3차원) RNN 모델 적용 
x_train = x_train.reshape(-1,28,28)
x_test = x_test.reshape(-1,28,28)
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)

### y값 알아보기 (넘파이)
# print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)
# array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],dtype=int64))

##### 원핫인코더(분류) ==> 컴파일에서 sparse_categorical_crossentropy 로 해결

#2. 모델구성
model = Sequential()

model.add(Reshape(target_shape=(28,28,1), input_shape=(28,28))) #28,28,1
# Reshape layer로 차원 변경, (N, 28, 28, 10) # 4차원
model.add(Conv2D(64, (3,3), activation='relu'))  #26,26,64
model.add(MaxPooling2D())                        # 13,13,64
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu'))   #11,11,64

model.add(GlobalAveragePooling2D()) # 4차원 => 2차원 

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax'))  

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009


model.compile(loss="sparse_categorical_crossentropy", 
              optimizer=Adam(learning_rate=learning_rate),
              metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    verbose=1,
    restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5, # learning_rate(러닝레이트) 비율 조절

)

# import datetime
# date = datetime.datetime.now()
# date = date.strftime('%d%m-%H%M')

# path ='./_save/keras36/'

# filename = '{epoch:04d}-{val_loss:.4f}.keras'
# filepath = "".join([path, "k36_", date, "-", filename])

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     save_best_only=True,
#     verbose=1,
#     filepath=filepath,
#     mode='min',
# )

start_time=time.time()
model.fit(x_train,y_train,
          epochs=100, 
          batch_size=64, 
          verbose=1,
          validation_split=0.2,
          callbacks = [es, rlr],
          )
end_time=time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

print('loss : ', round(loss[0],3))
print('acc : ', round(loss[1],3))

print(x_test.shape) #(10000, 28, 28)
print(y_test.shape) #(10000,)

y_pred = model.predict(x_test)
print(y_pred.shape) #(10000, 10)

y_pred = np.argmax(y_pred, axis=1)
print(y_pred.shape) #(10000,)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)
print ('걸린시간 : ', round(end_time-start_time,2),'초' )

### 결과 5 GlobalAveragePooling2D
# loss :  0.24
# acc :  0.917
# acc_score :  0.9172
# 걸린시간 :  172.24 초

### DNN 3차
# loss :  0.489
# acc :  0.89
# acc_score :  0.8897
# 걸린시간 :  127.78 초

# ### LSTM 3차
# loss :  0.291
# acc :  0.908


