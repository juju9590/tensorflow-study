# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

# y=wd
# 2016.12.31 00:00 ~ 2017.01.01 00:00 (6*24)
#12.31 데이터는 잘라내고 
# x.shape(n,144,13), y.shape(n,144,1)

# import os
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" #메모리 모으기


import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error


#1. 데이터

start_data = time.time()

path="./_data/kaggle_jena/"

xy_data = pd.read_csv(path+'jena_climate_2009_2016.csv', index_col=0)
# print(xy_data.shape) #(420551, 14)

# y수정 :T (degC)
# rmse기준 : 1.48
y_col = xy_data[-144:]['T (degC)'] #예측지 정답 데이터
# print(y_col.shape) #(144,)

x_data = xy_data[:-288].drop(['T (degC)'], axis=1)
# print(x_data.shape) #(420263, 13)

y_data = xy_data[144:-144]['T (degC)']
# print(y_data.shape) #(420263,)

x_pred = xy_data.drop(['T (degC)'], axis=1)
x_pred = x_pred.iloc[ -288:-144, :]
print(x_pred.shape) #(144, 13)
# print(x_predict)

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

##### x_train 스케일링(2차원으로 변경하여 스케일링 한 후 3차원으로 원복) 
scaler = MinMaxScaler()

# 2차원으로 변환
x_train = x_train.reshape(-1, 13)
x_test = x_test.reshape(-1, 13)

print(x_train.shape, x_test.shape) #(48397824, 13) (12099456, 13)

# 스케일링 
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)  
print(np.min(x_train), np.max(x_train)) # 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test)) # 0.0 1.0000000000000004

# 3차원으로 원복
x_train = x_train.reshape(-1,144,13)
x_test = x_test.reshape(-1,144,13)

print(x_train.shape, x_test.shape) # (336096, 144, 13) (84024, 144, 13)

y_train = y_train.reshape(-1,144,1)
y_test = y_test.reshape(-1,144,1)

print(y_train.shape, y_test.shape) #(336096, 144, 1) (84024, 144, 1)

# 데이터 저장
data_path = "./_save/kaggle_jena1/"

# np.save(data_path + "jena_x_train_yyy.npy", arr=x_train)
# np.save(data_path + "jena_y_train_yyy.npy", arr=y_train)
# np.save(data_path + "jena_x_test_yyy.npy", arr=x_test)
# np.save(data_path + "jena_y_test_yyy.npy", arr=y_test)

# # 데이터 불러오기
x_train = np.load(data_path + "jena_x_train_yyy.npy")
y_train = np.load(data_path + "jena_y_train_yyy.npy")
x_test = np.load(data_path + "jena_x_test_yyy.npy")
y_test = np.load(data_path + "jena_y_test_yyy.npy")

# end_data = time.time()
# print('데이터 불러오기 :', round(end_data-start_data,2),'초')


#2. 모델구성

start_time=time.time()

model = Sequential()
# model.add(SimpleRNN(64, input_shape=(144, 13), return_sequences=True,)) # 행무시 열우선
model.add(LSTM(64, input_shape=(144, 13), return_sequences=True,))
model.add(LSTM(32, return_sequences=True,))
model.add(Dense(64))
model.add(Dense(32))

model.add(Dense(144))

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
    epochs=1, 
    batch_size=500,
    verbose=1,
    callbacks = [es, rlr ],
    )

# 전체 모델 저장
model_path = "./_save/kaggle_jena1/"
filename = 'jena_model_yyy.keras'

model.save(model_path + filename)

# 모델 불러오기
# model = load_model(model_path + filename)

end_time=time.time()
print('훈련 걸린시간 :',round(end_time-start_time, 2),'초')

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results) 
print(x_test.shape, y_test.shape) #


####
x_pred = split_x(x_pred, size) 
print(x_pred.shape) #(1, 144, 13)

x_pred = x_pred.reshape(1,144,13)
print(x_pred.shape) #(1, 144, 13)

y_pred = model.predict(x_pred)
print(y_pred.shape) #(1, 144, 144)

y_test = y_test[-144]
# print(y_test.shape) #(144, 1)

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)

print(' 결과 : ', y_pred)


