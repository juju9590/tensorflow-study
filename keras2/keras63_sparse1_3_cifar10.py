
# 41-3

# 실습 acc : 0.67 (n, 32,32,3)

from tensorflow.keras.datasets import cifar10

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
(x_train, y_train), (x_test, y_test)= cifar10.load_data()
# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

# print(np.max(x_train), np.min(x_train)) # 255 0
# print(np.max(x_test), np.min(x_test)) # 255 0

#### 스케일링 1
x_train = x_train/255. 
x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0 
# print(np.max(x_test), np.min(x_test)) # 1.0 0.0

x_train = x_train.reshape(-1,32*32*3)
x_test = x_test.reshape(-1,32*32*3)
# print(x_train.shape, x_test.shape) # (50000, 3072) (10000, 3072)

# exit()

##### y값 알아보기
# print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)
# array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

#2. 모델구성
model = Sequential()
model.add(Dense(512, input_shape=(32*32*3,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(10, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.005
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
          epochs=100, 
          batch_size=2048, 
          verbose=1,
          validation_split=0.2,
          callbacks = [es, rlr, mcp],
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


### GlobalAveragePooling2D
# loss :  0.67
# acc :  0.77
# acc_score :  0.7719
# 걸린시간 :  275.09 초
