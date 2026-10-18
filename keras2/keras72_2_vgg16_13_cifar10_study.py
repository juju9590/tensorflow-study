


# [실습] 비교
# 1. 이전에 본인이 한 최상의 결과보다 높을것
# 2. 가중치를 동결하고 훈련시켰을때 , trainable = False
# 3. 가중치를 동결하지 않고 훈련시켰을때 , trainable = True
# 4. 시간까지 비교


import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score


import tensorflow as tf
import random

random.seed(80)
np.random.seed(80)
tf.random.set_seed(80)

from keras.applications import VGG16 # 전이 모델 불러오기
from tensorflow.keras.datasets import cifar10

#1. 데이터
(x_train, y_train), (x_test, y_test)= cifar10.load_data()
# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

#### 스케일링 1
x_train = x_train/255.0
x_test = x_test/255.0


#02. 모델

vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3),
)

# 1번 (동결)
vgg16.trainable=False # 가중치 동결 (위 모델에 대한 훈련은 안됨, 추론만 가능)
#  Total params: 14,789,258 (56.42 MB)
#  Trainable params: 74,570 (291.29 KB)  ***************** fc 부분만 훈련된다
#  Non-trainable params: 14,714,688 (56.13 MB)  *************** 전이모델은 동결

# 2번 (비동결)
# vgg16.trainable=True 
# Total params: 14,789,258 (56.42 MB)
#  Trainable params: 14,789,258 (56.42 MB)
#  Non-trainable params: 0 (0.00 B)

# vgg16.summary()

model = Sequential()
model.add(vgg16)     # ======> 앞부분은 vgg16 모델을 그대로 가져온다.

### 커스터마이징 가능 부분 ####
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(32))
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
    epochs=30, 
    batch_size=64, 
    verbose=1,
    validation_split=0.2,
    # callbacks = [es, rlr, mcp],
    )
end_time=time.time()

# 모델 경로
model_path = './_save/keras_72_2/'
filename = 'trainable_False.keras'
# filename = 'trainable_True.keras'

#모델저장하기
model.save(model_path + filename)

# model = load_model(model_path + filename)ㅇ


# 4. 평가, 예측
print( "=============model.evaluate=================")
loss = model.evaluate(x_test, y_test, verbose=1)

print('loss : ', round(loss[0],3))
print('acc : ', round(loss[1],3))

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
# y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)
print ('모델/훈련 걸린시간 : ', round(end_time-start_time,3),'초' )

## RNN
# loss :  0.285
# acc :  0.9
# acc_score :  0.8997
# 걸린시간 :  1149.33 초

## vgg16.trainable=False (가중치 동결, 전이학습모델 사용)
# loss :  1.229
# acc :  0.584
# acc_score :  0.5836
# 모델/훈련 걸린시간 :  316.782 초 ===> 동결 모델은 시간이 오래걸림

## vgg16.trainable=True (가중치 동결X, 전이학습모델 사용)
# loss :  1.336
# acc :  0.807
# acc_score :  0.807
# 모델/훈련 걸린시간 :  1056.708 초 ===> 동결X 모델은 시간이 오래걸림



