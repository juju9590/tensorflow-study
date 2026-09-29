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

# exit()  00



# 데이터 저장
data_path = "./_save/kaggle_jena1/"

np.save(data_path + "jena_x_data.npy", arr=x_data)
np.save(data_path + "jena_y_data.npy", arr=y_data)

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
    epochs=50, 
    batch_size=500,
    verbose=1,
    callbacks = [es, rlr ],
    )

# 전체 모델 저장
model_path = "./_save/kaggle_jena1/"
filename = 'jena_model.keras'

model.save(model_path + filename)

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


# loss : 7540.70166015625
# (1, 144, 13)
# 1/1 [==============================] - 0s 269ms/step
# (144,)
#  결과 :  [118.641174 159.26411  167.4496   169.01741  169.34312  169.43044
#  169.45908  169.475    169.48276  169.48492  169.4906   169.49248
#  169.49158  169.49239  169.49344  169.49428  169.48291  169.45955
#  169.45593  169.46445  169.4754   169.48293  169.45651  169.45741
#  169.45824  169.46188  169.4754   169.48293  169.48293  169.46925
#  169.46445  169.45741  169.39404  169.41191  169.41321  169.37204
#  169.34416  169.32748  169.3422   169.27048  169.30846  169.26913
#  169.33365  169.30678  169.2271   169.1306   169.07327  169.06566
#  169.02647  169.06209  169.03654  169.0589   169.08215  169.0295
#  169.01413  169.05205  169.24948  169.21231  169.23009  169.30121
#  169.21156  169.32468  169.38037  169.44958  169.43126  169.46188
#  169.48233  169.49118  169.49567  169.49605  169.49605  169.49634
#  169.49608  169.49608  169.49608  169.49695  169.49608  169.49695
#  169.49608  169.49608  169.49608  169.49252  169.49608  169.49608
#  169.49252  169.49252  169.49252  169.49252  169.49252  169.49252
#  169.49252  169.49252  169.49252  169.49252  169.49252  169.49252
#  169.49252  169.49252  169.49252  169.49608  169.49608  169.49608
#  169.49608  169.49275  169.49275  169.49251  169.49158  169.48291
#  169.4625   169.4787   169.45764  169.29614  169.14073  169.38774
#  169.31944  169.25058  169.40225  169.46133  169.46188  169.48233
#  169.48293  169.46486  169.45125  169.41191  169.40657  169.25455
#  169.20905  169.20813  169.00143  169.01065  168.96298  168.9321
#  168.91612  168.93076  169.0505   169.10262  169.0601   168.9562
#  168.95352  168.9112   168.93077  168.977    169.00487  169.0316  ]

# loss : 5633.3681640625
# (1, 144, 13)
# 1/1 [==============================] - 1s 799ms/step
# (144,)
#  결과 :  [-11.752824 104.431114 111.81321  138.01746  139.62766  141.88943
#  142.21132  142.55835  142.70255  142.77966  142.85556  142.88437
#  142.90274  142.89812  142.9025   142.88066  142.86752  142.84755
#  142.82933  142.77724  142.75308  142.70834  142.66083  142.62392
#  142.57707  142.53302  142.5006   142.45004  142.41772  142.36235
#  142.3162   142.26628  142.22928  142.18639  142.15527  142.10445
#  142.07101  142.02701  141.98804  141.96933  141.92685  141.80087
#  141.91318  141.48839  141.67809  141.32373  141.14626  140.51247
#  140.14183  139.59721  139.16548  138.75912  138.46165  138.26862
#  138.08745  137.96423  137.86919  137.79008  137.82819  138.05908
#  138.63292  139.36867  139.75847  140.16235  140.32985  140.47453
#  140.57738  140.679    140.7407   140.79982  140.83287  140.88817
#  140.91537  140.93047  140.95131  140.91693  140.83109  140.80229
#  140.8033   140.77036  140.74884  139.95418  137.58032  136.38232
#  135.85423  135.79337  135.59152  134.7465   134.24945  133.65001
#  133.1818   132.75713  132.37111  132.0476   131.76839  131.50093
#  130.85745  130.53273  130.76122  130.95169  131.18044  131.4129
#  132.45319  134.74144  136.20175  137.06142  137.86185  138.41129
#  138.89973  139.25034  139.74597  139.95604  140.18327  140.39432
#  140.54405  140.67052  140.77225  140.86154  140.90857  140.96213
#  141.00212  141.03098  141.04364  141.08264  141.10803  141.11685
#  141.12799  141.13098  141.13828  141.13945  141.13963  141.14586
#  141.14615  141.15237  141.1533   141.15143  141.1534   141.15686
#  141.14725  141.1651   141.19997  141.16252  141.27892  141.3127  ]