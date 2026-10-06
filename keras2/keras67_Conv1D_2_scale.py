# 55-2 copy


import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

#1. 데이터
start_data = time.time()
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60],
              ])

y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x_predict = np.array([50,60,70])  #80 맞춰보아요.

print(x.shape) #(13, 3)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) #(13, 3, 1)

end_data = time.time()


#2. 모델구성
model = Sequential()
# model.add(LSTM(64, input_shape=(3, 1))) # 기본적으로 tanh로 -1~1사이

model.add(Conv1D(64, kernel_size=2, input_shape=(3,1)))
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Conv1D(64, 2, padding='same', activation='relu'))

# model.add(Flatten())
model.add(GlobalAveragePooling1D())

model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))

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

model.compile(loss="mse", 
              optimizer=Adam(learning_rate=learning_rate),
              )

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
    factor=0.1, # learning_rate(러닝레이트) 비율 조절
)

model.fit(x,y, 
          epochs=500, 
          batch_size=128,
          verbose=1,
          callbacks = [es, rlr ],
          )

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss :', results)

x_pred = x_predict.reshape(1,3,1)
y_pred = model.predict(x_pred)

print('[50,60,70]의 결과 : ', y_pred)


# 활성화함수 relu 적용 (learning_rate=0.01)
# loss : 0.46535298228263855
# [50,60,70]의 결과 :  [[79.10578]

# 활성화함수 빼고, units을 128/128/64/64/64 로 했을때 (learning_rate=0.01)
# loss : 0.677997887134552
# [50,60,70]의 결과 :  [[80.7382]]

# Conv1D
# loss : 0.00016125236288644373
# [50,60,70]의 결과 :  [[79.981415]]

# loss : 0.0028361077420413494
# [50,60,70]의 결과 :  [[80.00147]]




