# 50-2 copy
# 6만장에서 4만장 증폭해서 10만장 만들기

import numpy as np
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

#1. 데이터 
start_data=time.time()

from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape) #(10000, 28, 28) (10000,)

augment_size=40000

datagen = ImageDataGenerator(
    # rescale = 1./255,

    horizontal_flip=0.1,
    # vertical_flip=0.1,
    # width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    # zoom_range=0.1,
    # shear_range=0.1,
    fill_mode='nearest',
)

randidx = np.random.randint(x_train.shape[0], size=augment_size)
# x_train.shape[0]: 훈련 이미지의 총 장수
# size=argment_size: 뽑을 번호의 개수. 지금은 40,000개
# randidx: 뽑힌 이미지 번호들의 배열, shape (40000,)
print(randidx.shape) #(40000,)

x_aug = x_train[randidx].copy()
y_aug = y_train[randidx].copy()
# randidx에 적힌 번호의 사진들을 x_train에서 골라 새 배열 x_aug로 복사
# 이후 이 사진들을 증강
print(x_aug.shape,y_aug.shape) #(40000, 28, 28) (40000,)

# 3차원 데이터를 4차원으로 변환
x_aug = x_aug.reshape(
    x_aug.shape[0],    # 40000
    x_aug.shape[1],    # 28
    x_aug.shape[2],    # 28
    1,                 # 1             
)

print(x_aug.shape) # (40000, 28, 28, 1)

x_aug = next(datagen.flow(
    x_aug,
    batch_size=augment_size,
    shuffle=True,
))

print(x_aug.shape) #(40000, 28, 28, 1)
print(x_train.shape) #(60000, 28, 28) -> 4차원으로 맞춰주기

x_train = x_train.reshape(
    x_train.shape[0],
    x_train.shape[1],
    x_train.shape[2],
    1,
)
print(x_train.shape) #(60000, 28, 28, 1)

# x_aug, x_train 차원 맞췄으면 이제 데이터 붙여주기
x_train = np.concatenate((x_train, x_aug))/255.0
y_train = np.concatenate((y_train, y_aug))
# np.concatenate()는 합칠 배열들을 하나의 묶음(튜플 또는 리스트)으로 받음
# 외부괄호 = 함수, 내부괄호= 튜플

print(x_train.shape,y_train.shape ) #(100000, 28, 28, 1) (100000,)
print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)
# array([ 9919,  9995, 10056, 10022, 10056,  9991, 10008,  9926, 10004, 10023]))

# 원핫인코딩
ohe = OneHotEncoder(sparse_output=False)
y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

y_train=ohe.fit_transform(y_train)
y_test=ohe.transform(y_test)

print(y_train.shape, y_test.shape) # (100000, 10) (10000, 10)

end_data=time.time()


#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1))) 
model.add(MaxPool2D())
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu' )) 
model.add(Conv2D(32, kernel_size=(3,3), activation='relu' )) 
model.add(Conv2D(32, (3,3), activation='relu' )) 
model.add(MaxPool2D())
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu' )) 
model.add(Dropout(0.3))
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu' ))
# model.add(Flatten()) 
model.add(GlobalAveragePooling2D())

model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))

model.add(Dense(10, activation='softmax'))  

model.summary()

#3. 컴파일, 훈련

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'],
)

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
model.fit(x_train, y_train,
        epochs=1,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[es, mcp],
        )
end_time=time.time()


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss :", round(results[0],3))
print("acc :", round(results[1],3))

y_pred= np.argmax(model.predict(x_test), axis=1)
y_test= np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print("acc : ", round(acc,3))

print ('데이터 걸린시간 : ', round(end_data-start_data,2),'초' )
print ('훈련 걸린시간 : ', round(end_time-start_time,2),'초' )


# loss : 52.5
# acc : 0.778
# acc :  0.778







