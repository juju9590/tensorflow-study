# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

# y=wd
# 2016.12.31 00:00 ~ 2017.01.01 00:00 (6*24)
#12.31 데이터는 잘라내고 
# x.shape(n,144,13), y.shape(n,144,1)

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" #메모리 모으기


import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split

#1. 데이터

path="./_data/kaggle_jena/"

xy_data = pd.read_csv(path+'jena_climate_2009_2016.csv', index_col=0)
# print(xy_data.shape) #(420551, 14)

y_col = xy_data[-144:]['wd (deg)'] #예측지 정답 데이터
# print(y_col.shape) #(144,)

x_data = xy_data[:-288].drop(['wd (deg)'], axis=1)
# print(x_data.shape) #(420263, 13)

y_data = xy_data[144:-144]['wd (deg)']
# print(y_data.shape) #(420263,)

x_pred = xy_data.drop(['wd (deg)'], axis=1)
x_pred = x_pred.iloc[ -288:-144, :]
print(x_pred.shape) #(144, 13)
# print(x_predict)

# # 결측치 확인
# print(xy_data.info()) # 결측치 X
# print(xy_data.describe()) # 이상치 확인 가능 (봄~겨울 1~4)
# print(xy_data.isna().sum()) # 컬럼별 결측치 확인

start_split = time.time()

#### 시계열 데이터로 자르기
size = 144

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)


x_data = split_x(x_data, size) 
# print(x_data.shape) # (420120, 144, 13)
y_data = split_x(y_data, size) 
# print(y_data.shape) # (420120, 144)

end_split = time.time()
print("데이터 분리 : ", round(end_split-start_split, 2),"초")

#### train _ test _ split

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=0.8, random_state=333,
)

print(x_train.shape, y_train.shape) #(336096, 144, 13) (336096, 144)
print(x_test.shape, y_test.shape) #(84024, 144, 13) (84024, 144)

##### 스케일링을 위한 x_train reshape
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

print(x_train.shape, x_test.shape)

##### x, y data 스케일링 (0~1 로)
scaler = MinMaxScaler()
scaler.fit(x_data) 
x_data = scaler.transform(x_data)  

print(np.min(x_data), np.max(x_data)) #

# 데이터 저장
data_path = "./_save/kaggle_jena1/"

np.save(path + "jena_x_data.npy", arr=x_data)
np.save(path + "jena_y_data.npy", arr=y_data)

# # 데이터 불러오기
# x_data = np.load(data_path + "jena_x_data.npy")
# y_data = np.load(data_path + "jena_y_data.npy")


#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(64, input_shape=(144, 13), return_sequences=True,)) # 행무시 열우선
model.add(LSTM(64, input_shape=(144, 13), return_sequences=True,))
model.add(LSTM(32, return_sequences=True,))
model.add(Dense(64))
model.add(Dense(32))

model.add(Dense(1))

model.summary()

#3. 컴파일

from tensorflow.keras.optimizers import Adam

# learning_rate = 0.1
# learning_rate = 0.01
learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.00005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(
    loss="mse", 
    optimizer=Adam(learning_rate=learning_rate),
)

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=40,
    verbose=1,
    restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.1, # learning_rate(러닝레이트) 비율 조절
)

model.fit(x_train, y_train, 
    epochs=100, 
    batch_size=128,
    verbose=1,
    callbacks = [es, rlr ],
    )

# 전체 모델 저장
model_path = "./_save/kaggle_jena1/"
filename = 'jena_model.keras'

model.save(path + filename)

# 모델 불러오기
# model = load_model(model_path + filename)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results)

####
x_pred = split_x(x_pred, size) 
print(x_pred.shape) #(1, 144, 13)

x_pred = x_pred.reshape(1,144,13)
# print(x_pred.shape) #(1, 144, 13)

y_pred = model.predict(x_pred)
y_pred = y_pred.reshape(-1)
print(y_pred.shape) #(144,)

print(' 결과 : ', y_pred)

