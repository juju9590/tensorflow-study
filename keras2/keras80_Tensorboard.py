"""

pip install setuptools==68.2.2

커맨드창에서 tensorboard의 log_dir 경로 따라 들어간 뒤,
tensorboard --logdir=.

경로 안 따라간다면
tensorboard --logidr=c:/study/_save/tensorboard/_keras80


"""

# 36-3 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

###### 스케일링 1
x_train = x_train/255.
x_test = x_test/255

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

######## One-hot Encoding ######
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(-1, 1)
y_train = encoder.fit_transform(y_train)

y_test = y_test.reshape(-1,1)
y_test = encoder.fit_transform(y_test)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (5,5), input_shape=(28, 28, 1)))   # (24, 24, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu')) # (20,20,64). 파라미터 이름도 적어봄
model.add(Dropout(0.2))
model.add(Conv2D(32, (5,5), activation='relu')) # (8,8,32)
model.add(Dropout(0.2))
model.add(Flatten())    # (512,)
model.add(Dense(units=32, activation='relu'))   # units: Dense의 output 갯수
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(10, activation='softmax'))  # (10,)

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import TensorBoard

tb = TensorBoard(
    log_dir='c:/study/_save/tensorboard/_keras80/',
    histogram_freq=0,
    write_graph=True,
    write_images=True,
    )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=40,
    restore_best_weights=True
)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'])
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=2000,
    batch_size=128,
    verbose=1,
    validation_split=0.2,
    callbacks=[es, tb]
)
end_time = time.time()

#4. 평가, 예측
print("=========model.evaluate==========")
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("acc : ", loss[1])

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)

print("accuracy_score : ", acc)
print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results
# Epoch 47/2000
# 375/375 [==============================] - 4s 11ms/step - loss: 0.0122 - acc: 0.9963 - val_loss: 0.0703 - val_acc: 0.9900
# =========model.evaluate==========
# 313/313 [==============================] - 1s 3ms/step - loss: 0.0360 - acc: 0.9905
# loss :  0.035990845412015915
# acc :  0.9904999732971191
# 313/313 [==============================] - 1s 2ms/step
# accuracy_score :  0.9905
# 걸린 시간 :  204.53 초