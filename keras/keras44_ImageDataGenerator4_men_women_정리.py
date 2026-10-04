# 남자_여자 분류하기

import numpy as np
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D,Dense,Flatten,GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#1. 데이터

start_data = time.time()
'''
datagen = ImageDataGenerator(
    rescale=1./255,
)

data_path ='./_data/image/men_women/'

xy = datagen.flow_from_directory(
    data_path, 
    target_size=(150,150),
    batch_size=30000,
    class_mode='binary',
    color_mode='rgb',
    shuffle= True,
)
# Found 27167 images belonging to 2 classes.
x, y = xy[0] #x,y 첫번째 배치 한번에 꺼내기, 클래스는 배치단위로 제공
print(x.shape, y.shape) #(27167, 150, 150, 3) (27167,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.3,
    random_state=999,
    shuffle=True,
    stratify=y, #y 클래스의 비율에 맞춰 분리하다
)
print(x_train.shape, x_test.shape) #(19016, 150, 150, 3) (8151, 150, 150, 3)
print(y_train.shape, y_test.shape) #(19016,) (8151,)
''' 

# 데이터 저장 경로
data_path = './_save/image/men_women/'
data_filename = 'men_women_1004_'

# # 데이터 저장하기
# np.save(data_path + data_filename + 'x_train.npy', arr=x_train)
# np.save(data_path + data_filename + 'y_train.npy', arr=y_train)
# np.save(data_path + data_filename + 'x_test.npy', arr=x_test)
# np.save(data_path + data_filename + 'y_test.npy', arr=y_test)

# 데이터 불러오기
x_train = np.load(data_path + data_filename + 'x_train.npy')
y_train = np.load(data_path + data_filename + 'y_train.npy')
x_test = np.load(data_path + data_filename + 'x_test.npy')
y_test = np.load(data_path + data_filename + 'y_test.npy')

end_data = time.time()
print('data 걸린시간 :', round(end_data-start_data, 3),"초") 

#2. 모델 구성
start_model=time.time()
'''
model = Sequential()

model.add(Conv2D(128, (3,3), input_shape=(150,150,3))) # 행무시열우선 
model.add(MaxPool2D())
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(Conv2D(16,(3,3), activation='relu'))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
model.compile(
    loss = 'binary_crossentropy',
    optimizer ='adam',
    metrics=['acc'],
)
model.fit(
    x_train, y_train,
    epochs=1,
    batch_size=64,
    validation_split=0.2,
    shuffle=True,

)
'''

# 모델 저장 경로 및 이름
model_path ='./_save/image/men_women/'
filename = 'men_women_1004_'

# 전체 모델 저장하기 
# model.save(model_path + filename + ".keras" )

# 모델 불러오기
model = load_model(model_path + filename + ".keras")

end_model=time.time()
print("모델/훈련 걸린시간 :", round(end_model-start_model,2),"초")

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', round(results[0],3))
print('acc : ', round(results[1],3))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) #이진분류는 0 or 1로 분류, 반올림하여 1, 0으로 만들어 준다.

acc = accuracy_score(y_pred, y_test)
print("acc_score :", round(acc,3) )

# 결과
# data 걸린시간 : 107.913 초 (저장)
# 모델/훈련 걸린시간 : 338.19 초
# loss :  0.424
# acc :  0.797
# acc_score : 0.7972027972027972

# 결과 (불러오기)
# data 걸린시간 : 7.161 초 (불러오기)
# 모델/훈련 걸린시간 : 0.44 초
# loss :  0.424
# acc :  0.797
# acc_score : 0.797


