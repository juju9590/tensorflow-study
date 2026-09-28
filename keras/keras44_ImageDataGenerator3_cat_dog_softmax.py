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
start_data = time.time()

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
    class_mode='categorical',        # 다중분류 (이진분류도 다중분류로 할 수 있다.) 또한 이 형식에서는 원핫인코딩 자동으로 해결
    color_mode='rgb',     # 컬러
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150, 150),     
    batch_size=10000,              
    class_mode='categorical',        
    color_mode='rgb',     
    shuffle=False, # test에서 필요가 없다 
)
print(xy_test.class_indices) #{'cats': 0, 'dogs': 1}
exit()

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape) #(8005, 150, 150, 3) (8005, 2)
print(x_test.shape, y_test.shape)   #(2023, 150, 150, 3) (2023, 2)


# 데이터 저장하기
data_path ="./_save/image/cat_dog/"

np.save(data_path+'cat_dog_x_train.npy', arr=xy_train[0][0])
np.save(data_path+'cat_dog_y_train.npy', arr=xy_train[0][1])
np.save(data_path+'cat_dog_x_test.npy', arr=xy_test[0][0])
np.save(data_path+'cat_dog_y_test.npy', arr=xy_test[0][1])

# 데이터 불러오기
# x_train = np.load(data_path+'cat_dog_x_train.npy')
# y_train = np.load(data_path+'cat_dog_y_train.npy')
# x_test = np.load(data_path+'cat_dog_x_test.npy')
# y_test = np.load(data_path+'cat_dog_y_test.npy')

end_data = time.time()
print("데이터 걸린시간 : ", round(end_data-start_data,2),"초")


# 2. 모델구성

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(150,150,3), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (3,3), activation='relu'))

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc'],
              )

start_time = time.time()
model.fit(x_train, y_train,
          epochs= 50,
          batch_size=32,
          verbose=1,
          validation_split=0.2, 
          )
end_time = time.time()

# 전체 모델 저장
model_path ='./_save/image/cat_dog/'
filename = 'cat_dog_model_0928_2th.keras'

model.save(model_path + filename)

# model = load_model(model_path + filename)


# 4. 평가, 예측
results = model.evaluate(x_test, y_test,)
print('loss : ', round(results[0],3))
print('acc : ', round(results[1],3))

y_pred = np.argmax(model.predict(x_test), axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score ) 
print('훈련 걸린시간 : ', round(end_time-start_time,2), "초")



# softmax (epochs=50, batch_size=64, 모델 layer 단순화)
# 'cat_dog_model_0928.keras'
# loss :  0.457
# acc :  0.796
# acc_score :  0.7958477508650519
# 훈련 걸린시간 :  227.32 초

# softmax (epochs=50, batch_size=32, 모델 layer 단순화)
# 'cat_dog_model_0928_2th.keras'
# loss :  0.437
# acc :  0.84
# 64/64 [==============================] - 1s 10ms/step
# acc_score :  0.8403361344537815
# 훈련 걸린시간 :  264.66 초







