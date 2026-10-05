# 42-4 카피

import numpy as np # 수치 계산에 특화
import pandas as pd # sklearn 만큼 강력함
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM,GRU,SimpleRNN,Bidirectional
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
path = "D:\\tensorflow_study\\_data\\ddarung\\" # 집
# path = "./study/_data/ddarung/" # 학원

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "submission.csv", index_col=0)

# 데이터 받으면 shape 찍기
print(train_csv.shape) # (1459, 10) 
print(test_csv.shape) # (715, 9) 
print(submission.shape) # (715, 1)

#### 결측치 처리
train_csv = train_csv.dropna() # 결측치 있는 row 자체를 없애버림
print(train_csv) # [1328 rows x 10 columns]

# train_csv를 x와 y로 분리
x = train_csv.drop(['count'], axis=1) # 열(컬럼) 삭제 | 행0, 열1
print(x) # [1328 rows x 9 columns]

y = train_csv['count'] # pandas에서 컬럼만 빼는거
print(y.shape) # (1328,)

# 트레인, 테스트 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7, 
    random_state=333,
    )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train) # x_train 만 fit  적용

x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) #0.0 1.0
print(np.min(x_test), np.max(x_test)) #-0.01 1.07

print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
print(y_train.shape, y_test.shape) #(929,) (399,)

########### X 데이터를 RNN에 넣기 위해 3차원으로 변환
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

print(x_train.shape, x_test.shape) #(929, 9, 1) (399, 9, 1)
print(y_train.shape, y_test.shape) #(929,) (399,)

# 2. 모델 구성
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(9,1)))
model.add(LSTM(128, return_sequences=True, input_shape=(9,1)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(1,))  

model.summary()

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")

import time
start_time = time.time()
hist = model.fit(
    x_train, y_train, 
    epochs=500, 
    batch_size=160,
    validation_split=0.2,
                )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)
print("r2 : ", r2_score(y_test, y_pred))
print("rmse : ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("걸린시간 :", round(end_time-start_time,2), "초")

###### dnn >>> > cnn 1차
# loss :  1809.003662109375
# r2 :  0.7488118498116965
# rmse :  42.53238308731484

### CNN >> LSTM 
# loss :  1810.8421630859375
# r2 :  0.7485565732807784
# rmse :  42.55398992274758
# 걸린시간 : 87.66 초

