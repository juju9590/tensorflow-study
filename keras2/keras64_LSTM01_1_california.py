# 러닝메이트
# 27-1 copy

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import time


#1. 데이터
datasets = fetch_california_housing() 
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) 
x = scaler.transform(x) 
# print(x)
# print(np.min(x), np.max(x)) 

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,
    # test_size=0.2,
    # shuffle=True, #디폴트 섞는다
    random_state=777,
) 

print(x_train.shape) #(16512, 8)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
print(x_train.shape) #(16512, 8, 1)

print(x_test.shape) #(4128, 8)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print(x_test.shape) #(4128, 8, 1)


print(y_test.shape) #(4128,)


#2. 모델구성
model = Sequential()

model.add(LSTM(64, input_shape=(8, 1))) 
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))

model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate) )

start_time = time.time() 

hist = model.fit(x_train, y_train, 
                epochs=300, batch_size=32,
                validation_split=0.2
                )
end_time = time.time()

#4. 평가, 예측


loss = model.evaluate(x_test, y_test, ) 
print("loss :", loss)

print(x_test.shape) 

y_pred = model.predict(x_test)
print(y_pred.shape) #

print(y_test.shape) #(102,)

r2 = r2_score(y_test, y_pred)
print("r2 : ", round(r2,3))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", round(rmse,3))

print("걸린시간 :", round(end_time - start_time,2), "초")


##### learning_rate = 0.01
# loss : 0.3875690698623657
# r2 :  0.699
# rmse :  0.623
# 걸린시간 : 268.19 초

### 결과
# r2 :  0.687
# rmse :  0.635
# 걸린시간 : 305.92 초

