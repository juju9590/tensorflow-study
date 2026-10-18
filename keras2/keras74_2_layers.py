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
model.add(Dense(10, activation='softmax'))

model.summary()

print(len(model.weights)) # 30 = 13(vgg16)*2(w, b) + 2(Dense)*2(w, b)
print(len(model.trainable_weights)) # 4 = 2(Dense)*2(w, b)

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








