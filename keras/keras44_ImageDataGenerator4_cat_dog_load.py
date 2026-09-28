# 실습 acc : 0.77


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import time
import datetime
from sklearn.metrics import accuracy_score

# 1. 데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,            

    # horizontal_flip=True,       # 수평 뒤집기,
    # vertical_flip=True,         # 수직 뒤집기,
    # width_shift_range=0.1,      # 평형이동
    # height_shift_range=0.1,
    # rotation_range=5,           # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,             # 확대
    # shear_range=0.7,            # 좌표하나를 고정하고 다른 몇개의 좌표를 이동(한마디로 찌부) 
    # fill_mode='nearest',        # 변화나 이동으로 인해 없어진 값은 근처의 값으로 채운다
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

np_path = './_data/kaggle_cat_dog_npy/'

np.save(np_path + 'keras45_02_cat_dog_x_train.npy', arr=xy_train[0][0])  
np.save(np_path + 'keras45_02_cat_dog_y_train.npy', arr=xy_train[0][1])  
np.save(np_path + 'keras45_02_cat_dog_x_test.npy', arr=xy_test[0][0])  
np.save(np_path + 'keras45_02_cat_dog_y_test.npy', arr=xy_test[0][1])

# x_train = np.load(np_path + 'keras45_02_cat_dog_x_train.npy')
# y_train = np.load(np_path + 'keras45_02_cat_dog_y_train.npy')
# x_test = np.load(np_path + 'keras45_02_cat_dog_x_test.npy')
# y_test = np.load(np_path + 'keras45_02_cat_dog_y_test.npy')

print(x_train.shape, y_train.shape) #(8005, 150, 150, 3) (8005,)
print(x_test.shape, y_test.shape)   #(2023, 150, 150, 3) (2023,)

# exit()

# 실습 : acc 1.0
# 2. 모델구성

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(150,150,3), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()

# exit()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=100,
    restore_best_weights=True,
    verbose=1,
)

date = datetime.datetime.now()
date = date.strftime('%d%m-%H%M')

path ='./_save/keras44/'

filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "keras44_cat_dog_save_model.keras", date, "-", filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath,
    verbose=1,
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs= 50,
          batch_size=100,
          verbose=1,
          validation_split=0.2, 
        #   callbacks=[es,mcp],         
          )
end_time = time.time()


##### 모델 저장
path ='./_save/keras44_model/'
# filename = 'keras44_8_save_model_{epoch:04d}-{val_loss:.4f}.keras'
model.save(path + 'keras44_cat_dog_save_model.keras') # 모델 전체 저장


# 4. 평가, 예측
results = model.evaluate(x_test, y_test,)
print('loss : ', round(results[0],3))
print('acc : ', round(results[1],3))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 반올림 처리

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score ) 
print('걸린시간 : ', round(end_time-start_time,3), "초")


# 결과 6
# loss :  3.146
# acc :  0.784
# acc_score :  0.7839841819080573
# 걸린시간 :  628.565 초









