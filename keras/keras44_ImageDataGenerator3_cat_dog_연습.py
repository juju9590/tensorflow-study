# 실습 acc : 0.77

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import time
import datetime
from sklearn.metrics import accuracy_score

# 1. 데이터 

# 데이터 수치화
train_datagen = ImageDataGenerator(
    rescale = 1/255.,

    horizontal_flip = True, 
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode='nearest',

)

test_datagen = ImageDataGenerator(
    rescale=1/255.
)

# 데이터 불러오기
path_train= './_data/image/cat_dog/training_set/'
path_test= './_data/image/cat_dog/test_set/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(150,150),
    batch_size=10000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150,150),
    batch_size=10000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델구성

model = Sequential()
model.add(Conv2D(64,(3,3), input_shape=(150,150,3)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Dropout(0.2)) # 과적합을 줄인다.
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPool2D())  # 속도 빠름, 연산량 줄음, 특징추출
model.add(Conv2D(64,(3,3), activation='relu'))

model.add(GlobalAveragePooling2D())
# model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# exit()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimazer='add', metrics=['acc'])

model.fit(x_train, y_train, 
          epochs=10,
          batch_size=100,
          verbose=1,
          validation_split=0.2,
          )

# 4. 평가, 훈련
results = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)



