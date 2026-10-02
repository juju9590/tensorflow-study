#28-3 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
from tensorflow.keras.datasets import boston_housing
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np


# 텐서플로우에서 데이터셋을 가져올때 아래와 같이 가져오면 된다

(x_train, y_train), (x_test, y_test)= boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train) 

x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test))

print(x_train.shape) #(404, 13)
print(x_test.shape) #(102, 13)
print(y_test.shape) #(102,)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print(x_train.shape) #(404, 13, 1)
print(x_test.shape) #(102, 13, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(13, 1))) 
model.add(LSTM(64, return_sequences=True,))
model.add(LSTM(32))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32))

model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    verbose=1,
    restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5, # learning_rate(러닝레이트) 비율 조절 : 0-1 사이

)

import time
start_time = time.time()

hist = model.fit(x_train, y_train, 
          epochs=500, 
          batch_size=128,
          validation_split=0.2,
          callbacks=[es, rlr,],
          )
end_time = time.time()
print("걸린시간 :", round(end_time-start_time,2), "초")

#4. 평가, 성능
loss = model.evaluate(x_test, y_test, )
print("loss :", loss)


print(x_test.shape) #(102, 13, 1)

y_pred = model.predict(x_test)
print(y_pred.shape) #(102, 1)

print(y_test.shape) #(102,)

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)

# ReduceLROnPlateau
# 걸린시간 : 13.34 초
# 22.047746658325195
# r2 :  0.7351426093842438
# rmse :  4.695503008599561

