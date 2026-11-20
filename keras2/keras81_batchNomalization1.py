#36-3 copy

import numpy as np
import pandas as pd
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test)= mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# print(np.max(x_train), np.min(x_train)) # 255 0
# print(np.max(x_test), np.min(x_test)) # 255 0

#### 스케일링 1
x_train = x_train/255. # .만 붙이면 float 형태로 출력하게 됨
x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0 (0~1 사이로 나옴)
# print(np.max(x_test), np.min(x_test)) # 1.0 0.0

##### X reshape  하는 이유 : input_shape 
x_train = x_train.reshape(-1,28,28,1) 
x_test = x_test.reshape(-1,28,28,1)
# print(x_train.shape ,x_test.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)

##### y값 알아보기

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))

##### 원핫인코더(분류)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False) 

# reshape 하는 이유.. 1차원을 2차원으로 바꿔줘야 한다. 
y_train = y_train.reshape(-1,1) # 데이터갯수를 안다면 y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

# print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

# exit()

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1))) # 26,26,64  // activation='linear' 생략된 상태
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu' )) #24,24,32
model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu' )) #23,23,32
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu' )) #22,22,16
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))  

model.summary()

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer='adam',
              metrics = ['acc'])


start_time=time.time()

model.fit(x_train,y_train,
          epochs=50, 
          batch_size=128, 
          verbose=1,
          validation_split=0.2,
          )
end_time=time.time()

# 4. 평가, 예측
print( "=============model.evaluate=================")
loss = model.evaluate(x_test, y_test, verbose=1)

print('loss : ', round(loss[0],2))
print('acc : ', round(loss[1],2))

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)
print ('걸린시간 : ', round(end_time-start_time,2),'초' )

##### 1차 - BatchNormalization 없을때
# 313/313 [==============================] - 0s 1ms/step
# loss :  0.07
# acc_score :  0.988
# 걸린시간 :  97.61 초

##### 2차 - activation 'relu' 이후 BatchNormalization (논문과 반대)
# 313/313 [==============================] - 0s 1ms/step
# loss :  0.04
# acc_score :  0.9924
# 걸린시간 :  108.02 초

##### 3차 - BatchNormalization 이후 activation 'relu' (논문방식)
# 313/313 [==============================] - 0s 1ms/step
# loss :  0.05
# acc_score :  0.987
# 걸린시간 :  106.74 초


















