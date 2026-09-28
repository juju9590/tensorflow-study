#28-3 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

#2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim=13))
model.add(Dense(7, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
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

# learning_rate로 적용되다가 어느 지점에서 최저점을 못찾고 핑퐁을 치면 
# reduceLR의 patience 만큼 참다가, factor만큼 조절된다 
# factor=0.5라면 learning_rate의 폭이 반으로 줄어든다... 
# 최저점을 찾을 수도 있다

import time
start_time = time.time()

hist = model.fit(x_train, y_train, 
          epochs=200, 
          batch_size=64,
          validation_split=0.2,
          callbacks=[es, rlr,],
          )
end_time = time.time()
print("걸린시간 :", round(end_time-start_time,2), "초")

# 훈련이 종료되면 마지막 W값이 정해진다.

print("=====================================")

#4. 평가, 성능
loss = model.evaluate(x_test, y_test, )
print(loss)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)

###### RobustScaler 
# 걸린시간 : 11.13 초
# 21.92060661315918
# r2 :  0.7366699474761296
# rmse :  4.681944788547605


# learning_rate = 0.01
# 걸린시간 : 5.74 초
# 21.37430191040039
# r2 :  0.7432326678015004
# rmse :  4.623234896299893

# ReduceLROnPlateau
# 걸린시간 : 13.34 초
# 22.047746658325195
# r2 :  0.7351426093842438
# rmse :  4.695503008599561

