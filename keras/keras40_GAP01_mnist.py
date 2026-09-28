#36-3 copy

import numpy as np
import pandas as pd
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

##### X reshape  하는 이유 : input_shape 
x_train = x_train.reshape(-1,28,28,1) 
x_test = x_test.reshape(-1,28,28,1)
print(x_train.shape ,x_test.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)

##### y값 알아보기

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))

##### 원핫인코더(분류)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False) 

y_train = y_train.reshape(-1,1) # 데이터갯수를 안다면 y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)


#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1))) 
model.add(Conv2D(64, (3,3), activation='relu' )) 
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu' ))
model.add(MaxPooling2D()) 
model.add(Conv2D(32, (3,3), activation='relu' )) 
model.add(Dropout(0.3))
model.add(Conv2D(32, (2,2), activation='relu' )) 
model.add(Dropout(0.3))
model.add(Conv2D(32, (2,2), activation='relu' ))  

# model.add(Flatten()) # 한마디로 reshape
# 특화된 이미지를 옆으로 쫙 피는 작업, 중심부에 강력한 특징 있음 
# 10만개 20만개에서 우리가 찾는값은 10개, 100개 정도
# Flatten에서 파라미터 확 튀다 
model.add(GlobalAveragePooling2D())

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation='relu'))

model.add(Dense(10, activation='softmax'))  

model.summary()

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer='adam',
              metrics = ['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    verbose=1,
    patience=20,
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
          epochs=50, 
          batch_size=128, 
          verbose=1,
          validation_split=0.2,
          callbacks = [es, mcp],
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

##### CPU 결과
# =============model.evaluate=================
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - acc: 0.9912 - loss: 0.0371     
# loss :  0.03710708022117615
# acc :  0.9911999702453613
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step  
# acc_score :  0.9912
# 걸린시간 :  595 초

##### GPU 결과
# loss :  0.05
# acc :  0.99
# 313/313 [==============================] - 0s 1ms/step
# acc_score :  0.9899
# 걸린시간 :  136.21 초

##### 성능향상 작업 : 0.995 맞추기
## 1) EarlyStopping 추가 
# loss :  0.05
# acc :  0.99
# acc_score :  0.9908
# 걸린시간 :  114.42 초

## 2) 커널사이즈 변경 + batch_size=32 설정 (Total params: 232,906 --> 177,098  ) ******
# loss :  0.04
# acc :  0.99
# acc_score :  0.992
# 걸린시간 :  244.61 초

## 3) batch_size=256, CNN Dropout을 model.add(Dropout(0.5))로 변경, epoch = 60 증가
# loss :  0.04
# acc :  0.99
# acc_score :  0.9902
# 걸린시간 :  144.69 초

### 4) 4번째 레이어층 필터 32로 향상, Dropout 0.3 변경
# loss :  0.03
# acc :  0.99
# acc_score :  0.9927
# 걸린시간 :  122.84 초

### 4) 2번째 레이어층 필터 128로 향상, Dropout 0.5 변경
# loss :  0.03
# acc :  0.99
# acc_score :  0.9927
# 걸린시간 :  122.84 초

### 5) MaxPooling 추가
# loss :  0.02
# acc :  0.99
# acc_score :  0.9935
# 걸린시간 :  156.29 초

### 6) GlobalAveragePooling2D
# loss :  0.02
# acc :  0.99
# acc_score :  0.9942
# 걸린시간 :  180.95 초















