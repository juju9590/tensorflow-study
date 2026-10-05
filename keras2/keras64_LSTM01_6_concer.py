# 42-6 카피

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional, Reshape
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_breast_cancer 

#1. 데이터
datasets = load_breast_cancer()
# print(datasets.DESCR) 
# print(datasets.feature_names) 

# 데이터 불러오기(1번)
# x = datasets.data
# y = datasets.target

# 데이터 불러오기(1번)
x = datasets['data'] 
y = datasets['target']

print(x.shape, y.shape) #(569, 30) (569,)
print('타입 : ', type(x)) 

print(y) #(y=범주) [0,1,0,0,...,1,1,1,0,1]

# 넘파이에서 y의 범주와 범주별 갯수 확인 방법
print("y의 범주 : ",np.unique(y)) # [0 1] 
print("y의 범주의 갯수 : ",np.unique(y, return_counts=True)) 
# (array([0, 1]), array([212, 357]))

# 판다스에서 y의 범주의 범주별 갯수 확인 방법 
# print(pd.DataFrame(y).value_counts()) 
# 1    357
# 0    212
# print(pd.Series(y).value_counts())
# 1    357
# 0    212

# train_test_spilt
x_train, x_test, y_train, y_test = train_test_split(
            x,y,
            random_state=908,
            test_size=0.3,
            shuffle=True,
            stratify=y, 
            )

# 스케일링
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train) 
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) #-2.3737740158538205 18.736137494346455
print(np.min(x_test), np.max(x_test)) #-1.8878923766816142 5.996470172961524

# 현재 차원 확인 
print(x_train.shape, x_test.shape) # (398, 30) (171, 30)
print(y_train.shape, y_test.shape) # (398,) (171,)

# X 데이터를 CNN에 넣기 위해 4차원으로 변환
x_train = x_train.reshape(-1,6,5,1)
x_test = x_test.reshape(-1,6,5,1)

print(x_train.shape, x_test.shape) #(398, 6, 5, 1) (171, 6, 5, 1)
print(y_train.shape, y_test.shape) #(398,) (171,)

# 다시 X 데이터를 RNN 넣기 위해 3차원으로 변환
x_train = x_train.reshape(-1,30,1)
x_test = x_test.reshape(-1,30,1)

print(x_train.shape, x_test.shape) #(398, 30, 1) (171, 30, 1)
print(y_train.shape, y_test.shape) #(398,) (171,)

#2. 모델구성
model = Sequential()

model.add(GRU(64, input_shape=(30, 1), activation='relu', return_sequences=True, )) #30,60
model.add(Bidirectional(LSTM(64, activation='relu',return_sequences=True, ))) #30,128
model.add(Bidirectional(LSTM(32, activation='relu',return_sequences=True, ))) #30,64
model.add(Reshape(target_shape=(64,6,5)))  # 4차원으로 변환 -> GAP 사용

# model.add(Flatten())
model.add(GlobalAveragePooling2D()) # 4차원  -> 2차원 출력

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid')) #(필수)이진분류모델

model.summary()

#3.컴파일, 훈련
model.compile(loss='binary_crossentropy', #(필수)이진분류모델
              optimizer='adam', # 아담이 그라디언트, 가중치 갱신 계산
              metrics=['acc'], 
              ) 


es = EarlyStopping(
    monitor="val_loss",
    mode='min',
    patience=20,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=150,
          batch_size=32,
          validation_split=0.2, #트레인에서 검증부분 분리
          verbose=1,
          callbacks=[es],
          )
end_time = time.time()
print("걸린시간 :", round(end_time-start_time,2), "초")


#4. 성능, 평가
loss = model.evaluate(x_test, y_test)
print("loss : ", round(loss[0],4)) # loss=binary_crossentropy
print("acc : ", round(loss[1],4))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 반올림 처리

print(x_test.shape)
print(y_pred.shape)
print(y_test.shape)

from sklearn.metrics import accuracy_score 

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score ) 



#### dnn >>>> cnn 1차 (cpu)
# loss :  0.0612
# accuracy :  0.9825
# acc_score :  0.9824561403508771
# 걸린시간 : 3.54 초

#### cnn >>> Rnn (cpu)
# 걸린시간 : 23.92 초
# loss :  0.1549
# acc :  0.9474
# acc_score :  0.9473684210526315


