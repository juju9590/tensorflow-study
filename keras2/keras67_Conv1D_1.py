# 54-1 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.layers import Conv1D, Flatten, GlobalAveragePooling1D

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) # 백터형 데이터

x = np.array([[1,2,3],
           [2,3,4],
           [3,4,5],
           [4,5,6],
           [5,6,7],
           [6,7,8],
           [7,8,9],
           ])        # 7,3 데이터로 분리 

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) #(7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)  #(7, 3, 1)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))  # 행무시, 열우선 (3,1)이 7개 있다..로 해석
# model.add(SimpleRNN(10, input_shape=(3, 1))) 
### 3차원으로 들어가서 1차원 또는 2차원으로 나옴 -> 바로 Dense와 연결가능

model.add(Conv1D(filters=16, kernel_size=2, input_shape=(3,1)))
model.add(Conv1D(32, 2))
model.add(Conv1D(16, 2))

model.add(Flatten())
# model.add(GlobalAveragePooling1D())

model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(
    loss='mse',
    optimizer='adam',
)

model.fit(x, y,
          epochs=100,
          batch_size=32,
          validation_split=0.2,
          )

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss :', results)

x_pred = np.array([8,9,10]).reshape(1,3,1)
y_pred = model.predict(x_pred)

print('[8,9,10]의 결과 : ', y_pred)



# loss : 23.871158599853516
# 1/1 [==============================] - 0s 89ms/step
# [8,9,10]의 결과 :  [[2.471821]
# [8,9,10]의 결과 :  [[10.860094]]

# Conv1D
# loss : 0.27348464727401733
# [8,9,10]의 결과 :  [[12.200713]]



