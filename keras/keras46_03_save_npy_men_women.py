# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset

# 데이터 Save


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import time
import datetime
from sklearn.metrics import accuracy_score

# 1. 데이터


# 데이터 수치화
start_data = time.time()
'''
datagen = ImageDataGenerator(
    rescale=1./255,
)

# 경로 
# path = './_data/image/men_women/'
path = 'D:\\tensorflow_study\\_data\\image\\men_women\\'

xy = datagen.flow_from_directory(
    path,
    target_size=(150, 150),
    batch_size=30000,          # 전체 데이터 수보다 크게
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)
# Found 27167 images belonging to 2 classes.
print(type(xy))
# <class 'keras.src.legacy.preprocessing.image.DirectoryIterator'>

x, y = xy[0] # 한번에 꺼내기
print(x.shape, y.shape) #(27167, 150, 150, 3) (27167,)
y= y.reshape(-1,1)
print(x.shape, y.shape) #(27167, 150, 150, 3) (27167, 1)

# x, y 꺼내기 (따로 꺼내기)
# x = xy[0][0]
# y = xy[0][1]
# print(x.shape, y.shape) #(27167, 150, 150, 3) (27167,)
# # y= y.reshape(-1,1)
# # print(x.shape, y.shape) #(27167, 100, 100, 3) (27167, 1)
# print(y.reshape(-1,1).shape)
# y의 쉐이프를 1차원에서 2차원으로 바꾸는 이유는 이진분류 모델의 출력이 Dense(1)이면
# 샘플당 출력도 하나라서 그 형태에 맞추려고 바꿈

# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder(sparse_output=False) 
# y = ohe.fit_transform(y)
# print(y, y.shape) # ... [0. 1.]] (27167, 2)

# train / test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=908,
    shuffle=True,
    stratify=y, # 클래스(y=정답) 비율을 유지하며 분리
)

print(x_train.shape, y_train.shape) # (21733, 100, 100, 3) (21733, 1)
print(x_test.shape, y_test.shape) # (5434, 100, 100, 3) (5434, 1)

# exit()
'''

# np_path = './_save/keras44/' 
np_path = './_save/image/men_women/'

# 저장 
# np.save(np_path + 'keras44_03_men_women_x_train.npy', arr=x_train) 
# np.save(np_path + 'keras44_03_men_women_y_train.npy', arr=y_train)
# np.save(np_path + 'keras44_03_men_women_x_test.npy', arr=x_test)
# np.save(np_path + 'keras44_03_men_women_y_test.npy', arr=y_test)

# 불러오기
x_train = np.load(np_path + 'keras44_03_men_women_x_train.npy')
y_train = np.load(np_path + 'keras44_03_men_women_y_train.npy')
x_test = np.load(np_path + 'keras44_03_men_women_x_test.npy')
y_test = np.load(np_path + 'keras44_03_men_women_y_test.npy')

end_data = time.time()
print("데이터 걸린시간 : ", round(end_data-start_data,3),"초")

# <데이터 변환 시간 >
# (27167, 150, 150, 3) (27167,)
# (27167, 150, 150, 3) (27167, 1)
# (21733, 150, 150, 3) (21733, 1)
# (5434, 150, 150, 3) (5434, 1)
# 데이터 걸린시간 :  81.014 초

# <데이터 저장 후 불러온 시간 >
# 데이터 걸린시간 :  7.636 초

# 2. 모델구성
start_time = time.time()
'''
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(150,150,3), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu'))

model.add(Flatten()) # 2, 3, 4 차원 받아서 2차원으로 펴주기
# model.add(GlobalAveragePooling2D()) # 4차원 받아서  2차원으로 펴주기

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid')) 

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam', 
              metrics=['acc'])


model.fit(x_train, y_train,
          epochs= 1,
          batch_size=128,
          verbose=1,
          validation_split=0.2,     
          )

'''
# 모델 저장
model_path = './_save/image/men_women/'
filename = 'men_women_first.keras'

# model.save(model_path + filename)

# 모델 불러오기 
model = load_model(model_path + filename)
end_time = time.time()
# 4. 평가, 예측
results = model.evaluate(x_test, y_test,)
print('loss : ', round(results[0],3))
print('acc : ', round(results[1],3))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 반올림 처리

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", round(acc_score,3) ) 
print('걸린시간 : ', round(end_time-start_time,3), "초")

### 결과
# 데이터 걸린시간 :  7.16 초 (불러오기)
# loss :  0.441
# acc :  0.793
# acc_score :  0.793
# 걸린시간 :  314.686 초 (epochs=1, cpu, 훈련시간)

### 결과
# 데이터 걸린시간 :  6.994 초
# loss :  0.441
# acc :  0.793
# acc_score :  0.793
# 걸린시간 :  0.799 초(epochs=1, cpu, 훈련 불러오기)