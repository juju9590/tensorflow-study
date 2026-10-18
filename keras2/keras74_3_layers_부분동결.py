import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(
    weights='imagenet',
    include_top = False,
    input_shape =(32,32,3),
)

vgg16.trainable =False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(90))
# model.add(Dense(80))
# model.add(Dense(70))
# model.add(Dense(60))
model.add(Dense(10, activation='softmax'))

model.layers[2].trainable = False # 특정 레이어만 동결 원함 

model.summary()

print(len(model.weights)) # 30 = 13(vgg16)*2(w, b) + 2(Dense)*2(w, b)
print(len(model.trainable_weights)) # 4 = 2(Dense)*2(w, b)

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  vgg16 (Functional)          (None, 1, 1, 512)         14714688  
                                                                 
#  flatten (Flatten)           (None, 512)               0         
                                                                 
#  dense (Dense)               (None, 100)               51300     
                                                                 
#  dense_1 (Dense)             (None, 90)                9090      
                                                                 
#  dense_2 (Dense)             (None, 10)                910       
                                                                 
# =================================================================
# Total params: 14,775,988
# Trainable params: 10,000
# Non-trainable params: 14,765,988
# _________________________________________________________________
# 32
# 4

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

#    Layer Type Layer Name  Layer Trainable
# 0  Functional      vgg16            False
# 1     Flatten    flatten             True
# 2       Dense      dense            False
# 3       Dense    dense_1             True
# 4       Dense    dense_2             True
# 5       Dense    dense_3             True
# 6       Dense    dense_4             True
# 7       Dense    dense_5             True








