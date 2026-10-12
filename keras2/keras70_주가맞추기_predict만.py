
# 삼성전자 1013 시가 맞추기
# 앙상블 쓸것
# LSTM, Conv1D 쓸것

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU, Input
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam


#1. 데이터

start_data = time.time()

### 데이터 저장
data_path = "./_save/keras70/"

# np.save(data_path + "samsung_x1_train.npy", arr=x1_train)
# np.save(data_path + "samsung_x2_train.npy", arr=x2_train)
# np.save(data_path + "samsung_x3_train.npy", arr=x3_train)
# np.save(data_path + "samsung_y_train.npy", arr=y_train)
# np.save(data_path + "samsung_x1_test.npy", arr=x1_test)
# np.save(data_path + "samsung_x2_test.npy", arr=x2_test)
# np.save(data_path + "samsung_x3_test.npy", arr=x3_test)
# np.save(data_path + "samsung_y_test.npy", arr=y_test)
# np.save(data_path + "samsung_x1_pred.npy", arr=x1_pred)
# np.save(data_path + "samsung_x2_pred.npy", arr=x2_pred)
# np.save(data_path + "samsung_x3_pred.npy", arr=x3_pred)

# # 데이터 불러오기
x1_train = np.load(data_path + "samsung_x1_train.npy")
x2_train = np.load(data_path + "samsung_x2_train.npy")
x3_train = np.load(data_path + "samsung_x3_train.npy")
y_train = np.load(data_path + "samsung_y_train.npy")
x1_test = np.load(data_path + "samsung_x1_test.npy")
x2_test = np.load(data_path + "samsung_x2_test.npy")
x3_test = np.load(data_path + "samsung_x3_test.npy")
y_test = np.load(data_path + "samsung_y_test.npy")
x1_pred = np.load(data_path + "samsung_x1_pred.npy")
x2_pred = np.load(data_path + "samsung_x2_pred.npy")
x3_pred = np.load(data_path + "samsung_x3_pred.npy")

end_data = time.time()
print(" data : ", round(end_data-start_data,3),"초")


#02. 모델구성

# 전체 모델 저장
model_path = "./_save/keras70/"
filename = 'samsung_keras70.keras'

# model.save(model_path + filename)

# 모델 불러오기
model = load_model(model_path + filename)

#4. 평가, 예측
results = model.evaluate([x1_test, x2_test, x3_test], y_test)
print('loss :', results) 

y_pred = model.predict([x1_pred, x2_pred, x3_pred])
print(y_pred)


