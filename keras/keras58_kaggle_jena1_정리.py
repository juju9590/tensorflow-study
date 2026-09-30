
# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

# 2016.12.31 00:10 ~ 2017.01.01 00:00 (6*24)
#12.31 데이터는 잘라내고
# x.shape(n,144,13), y.shape(n,144,1)

# import os
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" #메모리 모으기

# 작업순서
# 1. 데이터 : 원본데이터 가져오기 > x, y로 분리,


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
'''
### 원본 데이터 가져오기
path="./_data/kaggle_jena/"
xy_data = pd.read_csv(path+'jena_climate_2009_2016.csv', index_col=0)

### 원본 데이터에서 y정답, x, y 데이터 분리

y_col = xy_data[-144:]['T (degC)'] # 예측값 정답지
# print(y_col.shape) #(144,)
# print(type(y_col)) #<class 'pandas.core.series.Series'>
y_col = y_col.to_numpy()
# print(type(y_col)) #<class 'numpy.ndarray'>

x_data = xy_data[:-288].drop(['T (degC)'],axis=1) #x학습은 뒤에서 288행 제외
# print(x_data.shape) #(420263, 13)

y_data = xy_data[144:-144]['T (degC)'] #y학습은 앞에서 144행, 뒤에서 144행 제외
# print(y_data.shape) #(420263,)

x_predict = xy_data[-288:-144].drop(['T (degC)'],axis=1) #x_predict는 뒤에서 288~144행까지
# print(x_predict.shape) #(144, 13)

### train_test_split
### 1. shuffle=false (시간순서가 틀어지지 않게 false 조건 필수)
### 2. 시계열 데이터 분리 전 단계 진행 추천
### ㄴ 경계에서 학습 정답의 일부와 평가 입력의 일부가 겹치기 때문

x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_data,
    train_size=0.8,
    random_state=333,
    shuffle=False,              # 시간순유지, 섞이면 안됨
)

# print(x_train.shape, y_train.shape) # (336210, 13) (336210,)
# print(x_test.shape, y_test.shape) # (84053, 13) (84053,)

### 학습용 시계열 데이터 만들기
start_split=time.time()

size = 144

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

x_train = split_x(x_train, size)
x_test = split_x(x_test, size)
y_train = split_x(y_train, size)
y_test = split_x(y_test, size)
# print(x_train.shape) # (336067, 144, 13)
# print(x_test.shape) # (83910, 144, 13)
# print(y_train.shape) # (336067, 144)
# print(y_test.shape) # (83910, 144)

end_split=time.time()
# print("split_x 걸린시간:", round(end_split-start_split,2),"초")
# split_x 걸린시간: 216.92 초

##### x_train 스케일링 (3차원 ==> 2차원 변경 후 스케일링 ==> 3차원 원복)
scaler = MinMaxScaler()

# 2차원으로 변환
x_train = x_train.reshape(-1, 13)
x_test = x_test.reshape(-1, 13)

# print(x_train.shape, x_test.shape) #(48393648, 13) (12083040, 13)

# 스케일링(x_train, x_test)
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)  
# print(np.min(x_train), np.max(x_train)) # 0.0 1.0000000000000009
# print(np.min(x_test), np.max(x_test)) # -683.4586466165413 1.9473684210526312

### x_predict도 스케일링
x_predict = scaler.transform(x_predict)
# print(np.min(x_predict), np.max(x_predict)) # 0.002136752136752137 0.9861470998604744

# 3차원으로 원복(x_train, x_test)
x_train = x_train.reshape(-1,144,13)
x_test = x_test.reshape(-1,144,13)
# print(x_train.shape, x_test.shape) # (336067, 144, 13) (83910, 144, 13)

### x_predict도 3차원으로 변경
x_predict = x_predict.reshape(-1,144,13)
# print(x_predict.shape) # (1, 144, 13)

# x_train, x_tesx, x_predict 를 3차원으로 원복하는 이유는 RNN계열 모델에 적용하기 위해서..

### y_train, y_test 도  shape 맞추기
y_train = y_train.reshape(-1,144,1)
y_test = y_test.reshape(-1,144,1)
# print(y_train.shape) # (336067, 144, 1)
# print(y_test.shape) #(83910, 144, 1)
'''

### 데이터 저장
data_path = "./_save/kaggle_jena/"

# np.save(data_path + "jena_x_train_yyy.npy", arr=x_train)
# np.save(data_path + "jena_y_train_yyy.npy", arr=y_train)
# np.save(data_path + "jena_x_test_yyy.npy", arr=x_test)
# np.save(data_path + "jena_y_test_yyy.npy", arr=y_test)
# np.save(data_path + "jena_x_predict_yyy.npy", arr=x_predict)
# np.save(data_path + "jena_y_col_yyy.npy", arr=y_col)

# # 데이터 불러오기
x_train = np.load(data_path + "jena_x_train_yyy.npy")
y_train = np.load(data_path + "jena_y_train_yyy.npy")
x_test = np.load(data_path + "jena_x_test_yyy.npy")
y_test = np.load(data_path + "jena_y_test_yyy.npy")
x_predict = np.load(data_path + "jena_x_predict_yyy.npy")
y_col = np.load(data_path + "jena_y_col_yyy.npy")

#2. 모델구성
start_time=time.time()

model = Sequential()
# model.add(SimpleRNN(64, input_shape=(144, 13), return_sequences=True,)) # 행무시 열우선
model.add(LSTM(64, input_shape=(144, 13), return_sequences=True,))
model.add(LSTM(32, return_sequences=True,))
model.add(Dense(64))
model.add(Dense(32))

model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련

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
    factor=0.5, # learning_rate(러닝레이트) 비율 조절
)

model.fit(x_train, y_train, 
    epochs=1, 
    batch_size=500,
    verbose=1,
    callbacks = [es, rlr ],
    validation_split = 0.2,
    )

# 전체 모델 저장
model_path = "./_save/kaggle_jena/"
filename = 'jena_model_yyy.keras'

model.save(model_path + filename)

# 모델 불러오기
# model = load_model(model_path + filename)

end_time=time.time()
print('훈련 걸린시간 :',round(end_time-start_time, 2),'초')

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results) 

# 2016.12.31 00:10 ~ 2017.01.01 00:00 의 예측
y_pred = model.predict(x_predict)
# print(y_pred.shape) #(1, 144, 1)
y_pred = y_pred.reshape(-1)
# print(y_pred.shape) #(144,)

# y_col(정답지).shape = (144,)

r2 = r2_score(y_col, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_col, y_pred))
print("rmse : ", rmse)

##### y_pred를 csv로 저장하기
submission = pd.DataFrame()
submission['T (degC)'] = y_pred
# 빈 DataFrame에 'T (degC)'라는 컬럼을 만들고, 예측값을 순서대로 채우는 것
print(submission)

# submission 저장하기 
csv_path = "./_save/kaggle_jena/"

submission.to_csv(csv_path + "jena_predict.csv", index=False, )




