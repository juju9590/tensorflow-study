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

# VGG16 : weight(13) bias(13) + Dense : weight(2), bias (2)
print(len(model.trainable_weights)) # Dense : weight(2), bias (2)
