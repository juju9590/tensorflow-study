'''

전이학습 : Transfer Learning

Feature Extraction
- 사전 학습된 모델을 고정된 특징 추출기로 활용

pre-trainde Model
사전학습 모델

사전학습 모델 기반의 파인 튜닝(Fine-tuning)

'''


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential, Input
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import time
import datetime
from sklearn.metrics import accuracy_score

from keras.applications import VGG16 # pretrainde model


# 1. 데이터
np_path = './_save/keras44/'

# # 저장 
# # np.save(np_path + 'keras44_03_men_women_x_train.npy', arr=x_train) 
# # np.save(np_path + 'keras44_03_men_women_y_train.npy', arr=y_train)
# # np.save(np_path + 'keras44_03_men_women_x_test.npy', arr=x_test)
# # np.save(np_path + 'keras44_03_men_women_y_test.npy', arr=y_test)



# 불러오기
x_train = np.load(np_path + 'keras44_03_men_women_x_train.npy')
y_train = np.load(np_path + 'keras44_03_men_women_y_train.npy')
x_test = np.load(np_path + 'keras44_03_men_women_x_test.npy')
y_test = np.load(np_path + 'keras44_03_men_women_y_test.npy')

print(x_train.shape, y_train.shape) #(21733, 100, 100, 3) (21733, 1)
print(x_test.shape, y_test.shape) #(5434, 100, 100, 3) (5434, 1)

# 2. 모델구성

vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(100,100,3),
)

# 1번 (동결)
# vgg16.trainable=False 

# 2번 (비동결)
vgg16.trainable=True

model = Sequential()
model.add(Input(shape=(100, 100, 3)))

model.add(vgg16)     

### 커스터마이징 가능 부분 ####
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam', 
              metrics=['acc'])


start_time = time.time()
model.fit(x_train, y_train,
          epochs= 30,
          batch_size=64,
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
print("acc_score : ", round(acc_score,3)) 
print('걸린시간 : ', round(end_time-start_time,3), "초")

# 결과 2차 (GAP) 모델3
# loss :  0.278
# acc :  0.928
# acc_score :  0.928
# 걸린시간 :  662.285 초

#####  VGG16 동결모드 
# loss :  0.514
# acc :  0.349
# acc_score :  0.349
# 걸린시간 :  806.919 초

#####  VGG16 비동결모드 
# loss :  0.211
# acc :  0.349
# acc_score :  0.349
# 걸린시간 :  1191.861 초
