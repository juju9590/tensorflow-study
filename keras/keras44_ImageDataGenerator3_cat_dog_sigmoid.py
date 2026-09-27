# 실습 acc : 0.77


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import time
import datetime
from sklearn.metrics import accuracy_score

# 1. 데이터

start_data =time.time()

train_datagen = ImageDataGenerator(
    rescale=1./255,  
)          

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

# 파일의 경로
path_train = './_data/image/cat_dog/training_set/'  
path_test = './_data/image/cat_dog/test_set/'

xy_train = train_datagen.flow_from_directory(            
    path_train,                 # 폴더의 경로
    target_size=(150, 150),     # 이미지 크기 동일하게 맞추기
    batch_size=10000,             
    class_mode='binary',        # 이진분류
    color_mode='rgb',     # 흑백
    shuffle=True,
)
# Found 8005 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150, 150),     
    batch_size=10000,              
    class_mode='binary',        
    color_mode='rgb',     
    shuffle=False, # test에서 필요가 없다 
)
# Found 2023 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape) #(8005, 150, 150, 3) (8005,)
print(x_test.shape, y_test.shape)   #(2023, 150, 150, 3) (2023,)

# 데이터 저장하기 

data_path="./_save/image/cat_dog/"

np.save(data_path + 'cat_dog_sigmoid_x_train.npy', arr=xy_train[0][0])
np.save(data_path + 'cat_dog_sigmoid_y_train.npy', arr=xy_train[0][1])
np.save(data_path + 'cat_dog_sigmoid_x_test.npy', arr=xy_test[0][0])
np.save(data_path + 'cat_dog_sigmoid_y_test.npy', arr=xy_test[0][1])

# x_train = np.load(data_path + 'cat_dog_sigmoid_x_train.npy')
# y_train = np.load(data_path + 'cat_dog_sigmoid_y_train.npy')
# x_test = np.load(data_path + 'cat_dog_sigmoid_x_test.npy')
# y_test = np.load(data_path + 'cat_dog_sigmoid_y_test.npy')

end_data =time.time()


# 실습 : acc 1.0
# 2. 모델구성

model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', input_shape=(150,150,3), activation='relu'))
model.add(Conv2D(64, (3,3), strides=2, activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), strides=2, activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
# model.add(Dropout(0.2))

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()
model.fit(x_train, y_train,
          epochs= 1,
          batch_size=128,
          verbose=1,
          validation_split=0.2, 
          )
end_time = time.time()


##### 모델 저장
model_path ='./_save/image/cat_dog/'
filename = 'cat_dog_sigmoid_model.keras' # 확장자 : .keras

model.save(model_path + filename)
# model = load_model(model_path + filename)

# 4. 평가, 예측
results = model.evaluate(x_test, y_test,)
print('loss : ', round(results[0],3))
print('acc : ', round(results[1],3))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 반올림 처리

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score ) 

print('데이터 걸린시간 : ', round(end_time-start_time,3), "초")
print('훈련 걸린시간 : ', round(end_data-start_data,3), "초")


# 결과 6
# loss :  3.146
# acc :  0.784
# acc_score :  0.7839841819080573
# 걸린시간 :  628.565 초


### sigmoid (epochs= 1)
# Found 8005 images belonging to 2 classes.
# Found 2023 images belonging to 2 classes.
# (8005, 150, 150, 3) (8005,)
# (2023, 150, 150, 3) (2023,)
# loss :  0.693
# acc :  0.503
# acc_score :  0.5027187345526446
# 데이터 걸린시간 :  80.541 초
# 훈련 걸린시간 :  49.453 초











