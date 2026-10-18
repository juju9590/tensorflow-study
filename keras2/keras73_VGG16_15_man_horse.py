import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential, Input
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import time
import datetime
from sklearn.metrics import accuracy_score

import tensorflow as tf
import random

random.seed(333)
np.random.seed(444)
tf.random.set_seed(555)

from keras.applications import VGG16 # pretrainde model


# 1. 데이터
np_path = './_save/keras46_01/' # C:\study\_save\keras46_01

# 저장 
# np.save(np_path + 'keras46_01_horse_x_train.npy', arr=x_train) 
# np.save(np_path + 'keras46_01_horse_y_train.npy', arr=y_train)
# np.save(np_path + 'keras46_01_horse_x_test.npy', arr=x_test)
# np.save(np_path + 'keras46_01_horse_y_test.npy', arr=y_test)

# 불러오기
x_train = np.load(np_path + 'keras46_01_horse_x_train.npy')
y_train = np.load(np_path + 'keras46_01_horse_y_train.npy')
x_test = np.load(np_path + 'keras46_01_horse_x_test.npy')
y_test = np.load(np_path + 'keras46_01_horse_y_test.npy')

print(x_train.shape, x_test.shape) #(718, 150, 150, 3) (309, 150, 150, 3)
print(y_train.shape, y_test.shape) #(718, 2) (309, 2)



# 2. 모델구성
# vgg16 = VGG16()

# vgg16.summary()

# exit()



vgg16 = VGG16(
    # weights='imagenet',
    include_top=False,
    input_shape=(150,150,3),
)

# 1번 (동결)
# vgg16.trainable=False # 가중치 동결 (위 모델에 대한 훈련은 안됨, 추론만 가능)

# 2번 (비동결)
vgg16.trainable=True 


model = Sequential()
model.add(Input(shape=(150, 150, 3)))
model.add(vgg16)     

### 커스터마이징 가능 부분 ####
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam', 
              metrics=['acc'])


start_time = time.time()
model.fit(x_train, y_train,
          epochs= 78,
          batch_size=32,
          verbose=1,
          validation_split=0.2,     
          )
end_time = time.time()


# 4. 평가, 예측
results = model.evaluate(x_test, y_test,)
print('loss : ', round(results[0],3))
print('acc : ', round(results[1],3))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 반올림 처리

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score ) 
print('걸린시간 : ', round(end_time-start_time,3), "초")

#####  결과2 (에코프 50,  배치사이즈  32, flatten)
# loss :  0.014
# acc :  0.994
# acc_score :  0.9935275080906149
# 걸린시간 :  98.728 초

#####  VGG16 동결모드
# loss :  0.036
# acc :  0.984
# acc_score :  0.9838187702265372
# 걸린시간 :  63.506 초

#####  VGG16 비동결모드 (에코프 78)
# loss :  0.122
# acc :  0.984
# acc_score :  0.9838187702265372
# 걸린시간 :  223.681 초


