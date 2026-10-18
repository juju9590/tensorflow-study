# 44-1 카피

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential, Input
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D

import time
from sklearn.metrics import accuracy_score

from keras.applications import VGG16 


# 1. 데이터
start_data = time.time()

'''
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
'''

# 데이터 저장하기
data_path ="./_save/image/cat_dog/"

# np.save(data_path+'cat_dog_x_train.npy', arr=xy_train[0][0])
# np.save(data_path+'cat_dog_y_train.npy', arr=xy_train[0][1])
# np.save(data_path+'cat_dog_x_test.npy', arr=xy_test[0][0])
# np.save(data_path+'cat_dog_y_test.npy', arr=xy_test[0][1])

# 데이터 불러오기
x_train = np.load(data_path+'cat_dog_x_train.npy')
y_train = np.load(data_path+'cat_dog_y_train.npy')
x_test = np.load(data_path+'cat_dog_x_test.npy')
y_test = np.load(data_path+'cat_dog_y_test.npy')

print(x_train.shape, y_train.shape) #(8005, 150, 150, 3) (8005, 2)
print(x_test.shape, y_test.shape)   #(2023, 150, 150, 3) (2023, 2)

end_data = time.time()
print("데이터 걸린시간 : ", round(end_data-start_data,2),"초")


# 2. 모델구성

vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150,150,3),
)

# 1번 (동결)
# vgg16.trainable=False # 

# 2번 (비동결)
vgg16.trainable=True 

model = Sequential()
model.add(Input(shape=(150, 150, 3)))
model.add(vgg16)     # ======> 앞부분은 vgg16 모델을 그대로 가져온다.

### 커스터마이징 가능 부분 ####
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()
model.fit(x_train, y_train,
          epochs= 100,
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


# 결과
# 'cat_dog_model_0928_2th.keras'
# loss :  0.4365
# acc :  0.8403
# acc_score :  0.8403

#####  VGG16 동결모드
# loss :  1.568
# acc :  0.898
# acc_score :  0.897676717745922
# 걸린시간 :  1403.393 초

#####  VGG16 비동결모드 
# loss :  3.296
# acc :  0.902
# acc_score :  0.9016312407315867
# 걸린시간 :  3121.44 초

