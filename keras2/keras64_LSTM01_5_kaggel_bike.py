# https://www.kaggle.com/competitions/bike-sharing-demand/overview
# 52-5  카피

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, Dropout,GRU,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
# path = "./_data/kaggle_bike/"
path = "D:\\tensorflow_study\\_data\\kaggel_bike\\" #집


train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# 데이터 확인
print(train_csv) # [10886 rows x 11 columns]
print(test_csv) # [6493 rows x 8 columns]
print(submission) # [6493 rows x 1 columns]

# shape 확인
print(train_csv.shape) # (10886, 11)
print(test_csv.shape) # (6493, 8)
print(submission.shape) # (6493, 1)

# 데이터를 x, y로 분리 
x = train_csv.drop(['casual', 'registered', 'count'], axis=1) #열(컬럼) 삭제 
print(x.shape)# (8708, 8) (2178, 8)

y = train_csv['count']
print(y.shape) #(10886,)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.8, 
    random_state=142,
    )

# 스케일링
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) #-3.2189813545294523 5.852782306625195
print(np.min(x_test), np.max(x_test)) #-3.2189813545294523 5.852782306625195

print(x_train.shape, x_test.shape) #(8708, 8) (2178, 8)
print(y_train.shape, y_test.shape) #(8708,) (2178,)

# DNN >> RNN 변경, 2차원에서 3차원으로 reshape

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

print(x_train.shape, x_test.shape) #(8708, 8, 1) (2178, 8, 1)
print(y_train.shape, y_test.shape) #(8708,) (2178,)

# 2. 모델 구성
model = Sequential()

model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(8,1)))
model.add(GRU(64, activation='relu', return_sequences=True,))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', 
              optimizer=Adam(learning_rate=learning_rate),
              )

import time
start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=100, 
                 batch_size=32,
                 validation_split=0.2,
          )
end_time = time.time()

print("걸린시간 :", round(end_time-start_time,2), "초")


print("================== 학습 종료 ===================")

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)

# RobustScaler 적용  ############## ==> 성능개선

# Epoch 100/100
# 걸린시간 : 22.2 초
# loss :  22406.55078125
# r2 :  0.2988594174385071
# rmse :  149.68817197135184


# learning_rate = 0.01

# 걸린시간 : 33.52 초
# loss :  21901.94140625
# r2 :  0.3146495819091797
# rmse :  147.99303852926664

# dnn -> RNN
# 걸린시간 : 144.0 초
# loss :  21062.8203125
# r2 :  0.3409072160720825
# rmse :  145.13033608837608


