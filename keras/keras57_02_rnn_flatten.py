#57-1 카피
# Flatten 적용해서 1번과 성능비교

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60],
              ])

y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
# print(x.shape, y.shape) #(13, 3) (13,)

x_predict = np.array([50,60,70])  #80 맞춰보아요.

# exit()


#2. 모델구성
model = Sequential()

model.add(LSTM(units=64, input_shape=(3,1), return_sequences=True,))
model.add(LSTM(64, return_sequences=True,))
model.add(LSTM(32))

model.add(Flatten())

model.add(Dense(16))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련 
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.1
# learning_rate = 0.01
learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.00005
# learning_rate = 0.05
# learning_rate = 0.005

model.compile(
    loss="mse", 
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
    epochs=1000, 
    batch_size=32,
    verbose=1,
    callbacks = [es, rlr ],
    )

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss :', results)

x_pred = x_predict.reshape(-1,3,1)
y_pred = model.predict(x_pred)

print('[50,60,70]의 결과 : ', y_pred)


# 활성화함수 relu 적용 (learning_rate=0.01)
# loss : 0.46535298228263855
# [50,60,70]의 결과 :  [[79.10578]

# 활성화함수 빼고, units을 128/128/64/64/64 로 했을때 (learning_rate=0.01)
# loss : 0.677997887134552
# [50,60,70]의 결과 :  [[80.7382]]

# ======================

# loss : 0.0024867046158760786
# [50,60,70]의 결과 :  [[78.2657]]

# loss : 0.014092996716499329
# [50,60,70]의 결과 :  [[78.49143]]