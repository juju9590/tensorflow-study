# 데이터 증강 실습

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
start_data = time.time()

# 데이터 저장 경로
data_path = './_save/image/men_women/'
data_filename = 'men_women_1004_'

# 데이터 불러오기
x_train = np.load(data_path + data_filename + 'x_train.npy')
y_train = np.load(data_path + data_filename + 'y_train.npy')
x_test = np.load(data_path + data_filename + 'x_test.npy')
y_test = np.load(data_path + data_filename + 'y_test.npy')

# 전체 이미지의 수 (train 70%, test : 30%)
print(x_train.shape, x_test.shape) #(19016, 150, 150, 3) (8151, 150, 150, 3)
print(y_train.shape, y_test.shape) #(19016,) (8151,)

# 여자 사진의 수
x_train_woman = x_train[np.where(y_train >0)]
y_train_woman = y_train[np.where(y_train >0)]
x_test_woman = x_test[np.where(y_test >0)]
y_test_woman = y_test[np.where(y_test >0)]

print(x_train_woman.shape, y_train_woman.shape) #(6642, 150, 150, 3) (6642,)
# print(x_test_woman.shape, y_test_woman.shape) #(2847, 150, 150, 3) (2847,)

# # 남자 사진의 수
# x_train_man = x_train[np.where(y_train ==0)]
# y_train_man = y_train[np.where(y_train ==0)]
# x_test_man = x_test[np.where(y_test ==0)]
# y_test_man = y_test[np.where(y_test ==0)]

# print(x_train_man.shape, y_train_man.shape) #(12374, 150, 150, 3) (12374,)
# print(x_test_man.shape, y_test_man.shape) #(5304, 150, 150, 3) (5304,)

# 여자의 사진 이미지 증강 (+ 5700)
datagen_woman = ImageDataGenerator(
    horizontal_flip=True,
#     vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
#     zoom_range=0.1,
    rotation_range=5,
#     shear_range=0.1,
    fill_mode='nearest',
)

augment_size=5700
# x_train 기준 남자 12374, 여자 6642 이므로 여자 데이터 증강은  5700만 하면 됨

# 원본 사진 중  5700장의 인덱스 번호 랜덤 추출(randidx)
randidx=np.random.randint(x_train_woman.shape[0], size=augment_size)
# 해석 : randidx는 원본사진 6642장 중 랜덤으로 선정된 5700장의 인덱스 번호  
print(x_train_woman.shape[0]) # 6642
print(randidx.shape) # (5700,)

print(np.min(randidx), np.max(randidx)) # 2, 6641
# 랜덤으로 선정된  5700장은 원본 인덱스의 번호는 최소 1번 ~ 6640번 까지

x_aug_woman=x_train_woman[randidx].copy()
y_aug_woman=y_train_woman[randidx].copy()
# print(x_aug_woman.shape, y_aug_woman.shape) #(5700, 150, 150, 3) (5700,)

# 랜덤 인덱스로 뽑힌 원본 사진 5700장에 데이터 증강 처리 
xy_aug = datagen_woman.flow(
    x_aug_woman, y_aug_woman,
    batch_size=augment_size,
    shuffle=False,
)
# print(type(xy_aug)) #<class 'keras.src.legacy.preprocessing.image.NumpyArrayIterator'>
# print(len(xy_aug[0][0])) #5700


x_aug_woman, y_aug_woman = next(xy_aug) 
# print(x_aug_woman.shape, y_aug_woman.shape) # (5700, 150, 150, 3) (5700,)

# 원본 train과 데이터 증강 된 train 이미지 붙이기 
# concatenate = 괄호가 2번 (( a , b))
x_train = np.concatenate((x_train, x_aug_woman))
y_train = np.concatenate((y_train, y_aug_woman))

# print(x_train.shape, y_train.shape) #(24716, 150, 150, 3) (24716,)
# print(np.unique(y_train, return_counts=True)) #(array([0., 1.], dtype=float32), array([12374, 12342]))


end_data = time.time()
print('data 걸린시간 :', round(end_data-start_data, 3),"초")

# 데이터증강 포함 데이터 저장
data_aug_path = './_save/image/men_women/'
data_aug_filename = 'men_woman_aug_plus_'

np.save(data_aug_path + data_aug_filename + 'x_train.npy', arr=x_train)
np.save(data_aug_path + data_aug_filename + 'y_train.npy', arr=y_train)

# 최종 데이터 불러오기(경로 및 파일네임)
# data_aug_path = './_save/image/men_women/'
# data_aug_filename = 'men_woman_aug_plus_'
# data_path = './_save/image/men_women/'
# data_filename = 'men_women_1004_'

# x_train = np.load(data_aug_path + data_aug_filename + 'x_train.npy' )
# y_train = np.load(data_aug_path + data_aug_filename + 'y_train.npy' )
# x_test = np.load(data_path + data_filename + 'x_test.npy')
# y_test = np.load(data_path + data_filename + 'y_test.npy')

#2. 모델구성
start_model = time.time()

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=(150,150,3)))
model.add(MaxPool2D())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc'],
)

model.fit(
    x_train, y_train,
    epochs=1,
    batch_size=128,
    validation_split=0.2,
    shuffle=True,
    # callbacks=['es','mcp', 'rlr'],
)
end_model = time.time()
print('모델/훈련 걸린시간 :', round(end_model-start_model, 3),"초")

# 모델 저장 경로 및 이름
model_aug_path = './_save/image/men_women/'
model_aug_filename = 'men_womne_aug_model.keras'

# 전체 모델 저장
model.save(model_aug_path + model_aug_filename)

# 전체 모델 불러오기
# model = load_model(model_aug_path + model_aug_filename)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss : ", round(results[0],3))
print("acc : ", round(results[1],3))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_pred, y_test)
print("acc_score : ", round(acc_score,3))

# 결론
# data 걸린시간 : 70.325 초
# 모델/훈련 걸린시간 : 236.636 초
# loss :  0.397
# acc :  0.823
# acc_score :  0.823






