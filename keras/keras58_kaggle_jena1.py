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
from sklearn.metrics import r2_score,mean_squared_error


#1. 데이터

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
model.add(SimpleRNN(64, input_shape=(144, 13), return_sequences=True,)) # 행무시 열우선
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
    epochs=1, 
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
print(x_pred.shape) #(1, 144, 13)

y_pred = model.predict(x_pred)
y_pred = y_pred.reshape(-1)
print(y_pred.shape) #(144,)

y_test = y_test[-144]
y_test = y_test.reshape(-1)
print(y_test.shape) #(144,)

# exit()

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)

print(' 결과 : ', y_pred)


# loss : 12.702510833740234
# (1, 144, 13)
# 1/1 [==============================] - 0s 466ms/step
# (144,)
# (144,)
# r2 :  -806.2754892239806
# rmse :  22.149199657340894
#  결과 :  [ 0.46604496 -0.6353549  -1.6286683  -2.4741752  -3.0480483  -3.4889474
#  -3.890127   -4.246936   -4.551304   -4.814457   -5.0314064  -5.2077093
#  -5.3240886  -5.414759   -5.472055   -5.5466723  -5.6411276  -5.7388787
#  -5.8139343  -5.8823442  -5.9685335  -6.062557   -6.1608486  -6.206286
#  -6.2362013  -6.2462177  -6.2416954  -6.2233257  -6.1968307  -6.1689835
#  -6.1401653  -6.112185   -6.0858502  -6.062309   -6.0444098  -6.0601044
#  -6.076207   -6.0884414  -6.1387215  -6.190276   -6.244248   -6.278427
#  -6.347403   -6.422239   -6.4651303  -6.4751863  -6.4606385  -6.4401484
#  -6.4104056  -6.370209   -6.3254437  -6.2797356  -6.2357388  -6.196487
#  -6.203042   -6.2457914  -6.320598   -6.359939   -6.4201326  -6.427746
#  -6.475769   -6.5034595  -6.516874   -6.4882317  -6.446555   -6.3895006
#  -6.325117   -6.226753   -6.1976213  -6.239912   -6.2825685  -6.2494016
#  -6.248565   -6.320687   -6.404081   -6.4840417  -6.5570297  -6.5489254
#  -6.4539065  -6.331595   -6.2044086  -6.075162   -5.9467525  -5.8177214
#  -5.6744843  -5.515014   -5.369121   -5.242174   -5.1386366  -5.0311446
#  -4.8664174  -4.3025002  -3.5070915  -2.6420555  -1.901228   -1.6205688
#  -1.5781972  -1.8852339  -2.2522705  -2.6628358  -3.0813484  -3.4359598
#  -3.759198   -4.0282116  -3.9081929  -3.7159564  -3.6444404  -3.8429291
#  -4.071517   -4.2787805  -4.4573927  -4.6294117  -4.7879806  -4.9158583
#  -5.130658   -5.3974924  -5.644132   -5.855871   -6.0351176  -6.1879497
#  -6.317449   -6.4290557  -6.527142   -6.590045   -6.596114   -6.5305223
#  -6.401856   -6.282737   -6.2561235  -6.256806   -6.216408   -6.111622
#  -5.991683   -5.963613   -5.9647627  -5.9681926  -6.021168   -6.0336547
#  -6.0482574  -6.0547867  -6.087608   -6.169903   -6.3073273  -6.4054837 ]
# PS C:\study> 

