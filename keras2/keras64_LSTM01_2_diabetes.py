# 50-2 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Conv2D,SimpleRNN, GRU,Dropout, Bidirectional
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, MaxPool2D
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                train_size=0.8, 
                                random_state=333,
                                
                                )


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) #-2.357 3.304
print(np.min(x_test), np.max(x_test)) #-1.955 2.941

print(x_train.shape, x_test.shape) #(353, 10) (89, 10)
print(y_train.shape, y_test.shape) #(353,) (89,)

# RNN계열 모델 변경을 위해 2차원  -> 3차원으로 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
print(x_train.shape, x_test.shape) # (353, 10, 1) (89, 10, 1)

# 2. 모델 구성 (DNN -> LSTM으로 변경)
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(10,1)))
model.add(Bidirectional(LSTM(32, return_sequences=True, activation='relu',)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

import time
start_time = time.time()

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    restore_best_weights=True,
    patience=50,
    verbose=1,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5, # learning_rate(러닝레이트) 비율 조절

)

hist = model.fit(x_train, y_train, 
          epochs=500, 
          batch_size=64,
          validation_split=0.20,
          callbacks = [es, rlr],
          )

end_time = time.time()
print("걸린시간 :", round(end_time-start_time,2), "초")


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)


###### 결과
# loss :  2682.103515625
# r2 :  0.4941245923104354
# rmse :  51.78902641766279

###### 결과
# 걸린시간 : 23.43 초
# loss :  3003.34130859375
# r2 :  0.43353538105483314
# rmse :  54.80275073278241
