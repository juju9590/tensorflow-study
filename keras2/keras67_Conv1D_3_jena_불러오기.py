#58-1 copy (정리)
# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

# import os
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" #메모리 모으기

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D, Flatten

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error


#1. 데이터

data_start = time.time()
'''
### 원본 데이터 가져오기
path="./_data/kaggle_jena/"
xy_data = pd.read_csv(path+'jena_climate_2009_2016.csv', index_col=0)

### 원본 데이터에서 y정답, x, y 데이터 분리

y_col = xy_data[-144:]['T (degC)'] # 예측값 정답지
y_col = y_col.to_numpy()
print(y_col.shape) #(144,)

x_data = xy_data[:-288].drop(['T (degC)'],axis=1) 
y_data = xy_data[144:-144]['T (degC)'] 

x_predict = xy_data[-288:-144].drop(['T (degC)'],axis=1) 

### train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_data,
    train_size=0.8,
    random_state=333,
    shuffle=False,              # 시간순유지, 섞이면 안됨
)

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

##### x_train 스케일링 (3차원 ==> 2차원 변경 후 스케일링)
scaler = MinMaxScaler()

# 2차원으로 변환
x_train = x_train.reshape(-1, 13)
x_test = x_test.reshape(-1, 13)

# 스케일링(x_train, x_test)
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)  

### x_predict도 스케일링
x_predict = scaler.transform(x_predict)

# 3차원으로 원복(x_train, x_test)
x_train = x_train.reshape(-1,144,13)
x_test = x_test.reshape(-1,144,13)
print(x_train.shape, x_test.shape) # (336067, 144, 13) (83910, 144, 13)
print(y_train.shape, y_test.shape) #(336067, 144) (83910, 144)

# y_train = y_train.reshape(-1,144,1)
# y_test = y_test.reshape(-1,144,1)
# print(y_train.shape, y_test.shape) #(336067, 144, 1) (83910, 144, 1)

### x_predict도 3차원으로 변경
x_predict = x_predict.reshape(-1,144,13)
print(x_predict.shape) # (1, 144, 13)

'''
### 데이터 저장
data_path = "./_save/kaggle_jena/"
filename = "jena_Conv1D_"

# np.save(data_path + filename + "x_train.npy", arr=x_train)
# np.save(data_path + filename + "y_train.npy", arr=y_train)
# np.save(data_path + filename + "x_test.npy", arr=x_test)
# np.save(data_path + filename + "y_test.npy", arr=y_test)
# np.save(data_path + filename + "x_predict.npy", arr=x_predict)
# np.save(data_path + filename + "y_col", arr=y_col)

# # 데이터 불러오기
x_train = np.load(data_path + filename + "x_train.npy")
y_train = np.load(data_path + filename + "y_train.npy")
x_test = np.load(data_path + filename + "x_test.npy")
y_test = np.load(data_path + filename + "y_test.npy")
x_predict = np.load(data_path + filename + "x_predict.npy")
y_col = np.load(data_path + filename + "y_col.npy")


print(x_train.shape, x_test.shape) #(336067, 144, 13) (83910, 144, 13)
print(x_predict.shape) #(1, 144, 13)
print(y_train.shape, y_test.shape) #(336067, 144) (83910, 144) ==> 출력층 144

# 시계열 데이터 : Conv1D 현업에서 중요


data_end = time.time()
print("data_걸린시간 :", round(data_end-data_start,3),"초")

#2. 모델구성
start_time=time.time()
'''
model = Sequential()

model.add(Conv1D(64, 3, input_shape=(144,13), activation='relu'))
model.add(Conv1D(64, 3, activation='relu',))
model.add(MaxPool1D())
model.add(Conv1D(64, 3, activation='relu',))
model.add(LSTM(32, return_sequences=True,))

# model.add(Flatten())
model.add(GlobalAveragePooling1D())

model.add(Dense(64))
model.add(Dense(32))

model.add(Dense(144))

model.summary()


#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam

# learning_rate = 0.1
# learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
learning_rate = 0.0001
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
    epochs=100, 
    batch_size=64,
    verbose=1,
    callbacks = [es, rlr, ],
    validation_split = 0.2,
    )
'''
# 전체 모델 저장
model_path = "./_save/kaggle_jena/"
filename = 'jena_model_Conv1D.keras'

# model.save(model_path + filename)

# 모델 불러오기
model = load_model(model_path + filename)

end_time=time.time()
print('모델/훈련 걸린시간 :',round(end_time-start_time, 2),'초')

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results) 

y_pred = model.predict(x_predict)
y_pred = y_pred.reshape(-1)

print(y_col.shape)

r2 = r2_score(y_col, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_col, y_pred))
print("rmse : ", rmse)

exit()

# 결과 (Conv1D)
# data_걸린시간 : 3.424 초
# 모델/훈련 걸린시간 : 6.02 초
# loss : 7.087216377258301
# r2 :  0.1716616594083833
# rmse :  3.004137670587273
