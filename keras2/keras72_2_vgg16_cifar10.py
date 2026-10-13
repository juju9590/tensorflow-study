


# [실습] 비교
# 1. 이전에 본인이 한 최상의 결과.
# 2. 가중치를 동결하고 훈련시켰을때 , trainable = False
# 3. 가중치를 동결하지 않고 훈련시켰을때 , trainable = True
# 4. 시간까지 비교



import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import random

random.seed(80)
np.random.seed(80)
tf.random.set_seed(80)

from keras.applications import VGG16 # 전이 모델 불러오기
from tensorflow.keras.datasets import cifar10

#02. 모델

vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3),
)

# 1번
vgg16.trainable=False # 가중치 동결 (위 모델에 대한 훈련은 안됨, 추론만 가능)
#  Total params: 14,789,258 (56.42 MB)
#  Trainable params: 74,570 (291.29 KB)  ***************** fc 부분만 훈련된다
#  Non-trainable params: 14,714,688 (56.13 MB)  *************** 전이모델은 동결

# 2번
# vgg16.trainable=True 
# Total params: 14,789,258 (56.42 MB)
#  Trainable params: 14,789,258 (56.42 MB)
#  Non-trainable params: 0 (0.00 B)

# vgg16.summary()

model = Sequential()
model.add(vgg16)     # ======> 앞부분은 vgg16 모델을 그대로 가져온다.

### 커스터마이징 가능 부분 ####
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))

model.summary()



