import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

model = VGG16(
    weights='imagenet',
    include_top = False,
    input_shape =(32,32,3),
)

# model.trainable =False

model.layers[1].trainable = False # 특정 레이어만 동결 원함 (가져온 모델도 동결 가능)

model.summary()

print(len(model.weights)) # 26 = 13(vgg16)*2(w, b) 
print(len(model.trainable_weights)) # 24 = 12(vgg16)*2(w, b) , 1개 레이어는 동결 

import pandas as pd
pd.set_option('max_colwidth', None)
layers =[
    (type(layer).__name__ , layer.name, layer.trainable) #레이어 타입의 이름, 레이어의 이름, 레이어의 trainable 출력
    for layer in model.layers 
]

results = pd.DataFrame(
    layers, 
    columns=['Layer Type', 'Layer Name', 'Layer Trainable']
)
# 시리즈 : 백터형태, 데이터프레임  : 메트릭스 형태  ==> 판다스에는 딱 2종류 밖에 없음

print(results)

#       Layer Type    Layer Name  Layer Trainable
# 0     InputLayer       input_1             True
# 1         Conv2D  block1_conv1            False  # 특정 레이어만 동결(가져온 모델에서)
# 2         Conv2D  block1_conv2             True
# 3   MaxPooling2D   block1_pool             True
# 4         Conv2D  block2_conv1             True
# 5         Conv2D  block2_conv2             True
# 6   MaxPooling2D   block2_pool             True
# 7         Conv2D  block3_conv1             True
# 8         Conv2D  block3_conv2             True
# 9         Conv2D  block3_conv3             True
# 10  MaxPooling2D   block3_pool             True
# 11        Conv2D  block4_conv1             True
# 12        Conv2D  block4_conv2             True
# 13        Conv2D  block4_conv3             True
# 14  MaxPooling2D   block4_pool             True
# 15        Conv2D  block5_conv1             True
# 16        Conv2D  block5_conv2             True
# 17        Conv2D  block5_conv3             True
# 18  MaxPooling2D   block5_pool             True







