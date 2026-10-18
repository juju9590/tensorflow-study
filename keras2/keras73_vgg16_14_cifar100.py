# [실습] 비교
# 1. 이전에 본인이 한 최상의 결과보다 높을것
# 2. 가중치를 동결하고 훈련시켰을때 , trainable = False
# 3. 가중치를 동결하지 않고 훈련시켰을때 , trainable = True
# 4. 시간까지 비교


import time
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score


import tensorflow as tf
import random

random.seed(333)
np.random.seed(444)
tf.random.set_seed(555)

from keras.applications import VGG16 # 전이 모델 불러오기
from tensorflow.keras.datasets import cifar100

#1. 데이터
(x_train, y_train), (x_test, y_test)= cifar100.load_data()

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
# vgg16.trainable=False # 가중치 동결 (위 모델에 대한 훈련은 안됨, 추론만 가능)
#  Total params: 14,789,258 (56.42 MB)
#  Trainable params: 74,570 (291.29 KB)  ***************** fc 부분만 훈련된다
#  Non-trainable params: 14,714,688 (56.13 MB)  *************** 전이모델은 동결
# ㄴ 역전파가 안됨, 추론만 가능

# 2번 (비동결)
vgg16.trainable=True 
# Total params: 14,789,258 (56.42 MB)
#  Trainable params: 14,789,258 (56.42 MB)
#  Non-trainable params: 0 (0.00 B)
# ㄴ 재학습 필요할 경우

# vgg16.summary()

model = Sequential()
model.add(vgg16)     # ======> 앞부분은 vgg16 모델을 그대로 가져온다.

### 커스터마이징 가능 부분 ####
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(100, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss="sparse_categorical_crossentropy", 
              optimizer=Adam(learning_rate=learning_rate),
              metrics = ['acc'])

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
    factor=0.5,
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
    batch_size=256, 
    verbose=1,
    validation_split=0.2,
    callbacks = [es, rlr, mcp],
    )
end_time=time.time()

# 모델 경로
model_path = './_save/keras_73/'
filename = 'trainable_False.keras'
# filename = 'trainable_True.keras'

#모델저장하기
model.save(model_path + filename)

# model = load_model(model_path + filename)ㅇ


# 4. 평가, 예측
print( "=============model.evaluate=================")
loss = model.evaluate(x_test, y_test, verbose=1)

# print(y_test.shape) #(10000, 1)
# print(x_test.shape) #(10000, 32, 32, 3)

print('loss : ', round(loss[0],3))
print('acc : ', round(loss[1],3))

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
# y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)
print ('모델/훈련 걸린시간 : ', round(end_time-start_time,3),'초' )

## 결과 3 GlobalAveragePooling2D
# loss :  2.3
# acc :  0.4
# acc_score :  0.4011
# 걸린시간 :  360.63 초

## vgg16.trainable=False (가중치 동결, 전이학습모델 사용)
# loss :  2.685
# acc :  0.346
# acc_score :  0.3463
# 모델/훈련 걸린시간 :  186.906 초

## vgg16.trainable=True (가중치 동결X, 전이학습모델 사용)
# loss :  2.652
# acc :  0.335
# acc_score :  0.0112
# 모델/훈련 걸린시간 :  814.63 초
# 
# 모델/훈련 걸린시간 :  832.596 초
# ===========================================
# 모델명 : vgg19 loss : nan acc : 0.009999999776482582

# 모델/훈련 걸린시간 :  946.685 초
# ===========================================
# 모델명 : resnet50 loss : nan acc : 0.009999999776482582

# 모델/훈련 걸린시간 :  1470.67 초
# ===========================================
# 모델명 : resnet101 loss : nan acc : 0.009999999776482582

# 모델/훈련 걸린시간 :  57963.647 초
# ===========================================
# 모델명 : densenet201 loss : nan acc : 0.009999999776482582

# 모델/훈련 걸린시간 :  617.089 초
# ===========================================
# 모델명 : mobilenetv2_1.00_224 loss : nan acc : 0.009999999776482582

# 모델/훈련 걸린시간 :  787.075 초
# ===========================================
# 모델명 : efficientnetb0 loss : nan acc : 0.009999999776482582

