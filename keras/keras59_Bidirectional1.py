# 54-3 copy
# Bidirectional : 양방향 모델

# Please initialize `Bidirectional` layer with a `tf.keras.layers.Layer` instance. Received: 10
# initialize : 초기화 
# Bidirectional 자체 모델이 아니다. 랩핑 모델을 넣어주어야 한다. 주로 RNN 안에 넣어준다

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.array([[1,2,3],
           [2,3,4],
           [3,4,5],
           [4,5,6],
           [5,6,7],
           [6,7,8],
           [7,8,9],
           ])        # 7,3 데이터로 분리 

y = np.array([4,5,6,7,8,9,10])

# print(x.shape, y.shape) #(7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)  ### ([[[1],[2],[3]], ... [[[7],[8],[9]]]) 형식으로 바꿔줌
# print(x.shape)  #(7, 3, 1)

#2. 모델구성
model = Sequential()

# model.add(SimpleRNN(10, input_shape=(3, 1))) 
model.add(Bidirectional(SimpleRNN(10), input_shape=(3, 1)))
### simpleRNN을 양방향으로 2번 해라 
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

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
# learning_rate = 0.009

model.compile(loss="mse", 
              optimizer=Adam(learning_rate=learning_rate),
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=40,
    verbose=1,
    restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=10,
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

x_pred = np.array([8,9,10]).reshape(1,3,1)
y_pred = model.predict(x_pred)

print('[8,9,10]의 결과 : ', y_pred)



# [8,9,10]의 결과 :  [[10.860094]]

# loss : 3.4931952541228384e-05
# [8,9,10]의 결과 :  [[10.573267]]
