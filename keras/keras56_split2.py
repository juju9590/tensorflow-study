import numpy as np

import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0],
              ]).T                     #feature=2열
print(a.shape) #(10, 2)

size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)

# print(bbb)
# print(type(bbb)) #<class 'numpy.ndarray'>

print(bbb.shape) # (6, 5, 2) => RNN 3차 (batch=6, timestep=5, feature=2)2열
print("batch" ,bbb.shape[0])    #batch 6
print("timestep", bbb.shape[1]) #timestep 5
print("feature",bbb.shape[2])   #feature 2

# bbb[배치,타임스텝(행), 컬럼(열)] #3차원일 경우도 동일하게, 
# x 데이터에서 컬럼이 1이면 생략가능

# x_data = bbb[:, :4]
# x_data = bbb[:, :-1, :]  # [모든행, 마지막 전까지, 모든컬럼]
x_data = bbb[:, :-1] # [모든행, 마지막 전까지] 모든컬럼은 생략해도 된다.

# x_data = bbb[:, :(bbb.shape[1]-1)]
print(x_data.shape) #(6, 4, 2)
print("x_data:", x_data)

# y_data = bbb[:, -1, 0] # [ 5  6  7  8  9 10]
# y_data = bbb[:, -1, 1] # [5 4 3 2 1 0]
y_data = bbb[:, -1, -1] # [5 4 3 2 1 0] [모든행, 마지막행, 두번째 열]

# y_data = bbb[:, 4]
# y_data = bbb[:, (bbb.shape[1]-1)]
# print("y_data_1:", y_data)
# y_data = y_data.T
# print("y_data_2:", y_data)
# y_data = y_data[1]
# print("y_data_3:", y_data)

# y_data = bbb[:, 4].T[1]
# y_data = bbb[:, (bbb.shape[1]-1)].T[1] #y_data: [5 4 3 2 1 0] 
# y_data = bbb[:, (bbb.shape[1]-1)].T[0] #y_data: [ 5  6  7  8  9 10]

print(y_data.shape) #(6,)
print("y_data:", y_data) 

exit()

#2. 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(4, 2))) # x 특성이 4개
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))

model.add(Dense(1))

model.summary()

#3. 컴파일

from tensorflow.keras.optimizers import Adam

# learning_rate = 0.1
# learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.00005
learning_rate = 0.05
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

model.fit(x_data,y_data, 
          epochs=500, 
          batch_size=128,
          verbose=1,
          callbacks = [es, rlr ],
          )

#4. 평가, 예측
results = model.evaluate(x_data, y_data)
print('loss :', results)

x_predict =np.array([[7,3],[8,2],[9,1],[10,0]])
x_pred = x_predict.reshape(1,4,2)
y_pred = model.predict(x_pred)

print('[[7,3],[8,2],[9,1],[10,0]]의 결과 : ', y_pred)

# y_data: [5 4 3 2 1 0] 일때 예측값 (1개)
#  결과 (learning_rate : 0.001)
#  loss : 7.994377665454522e-05
# [[7,3],[8,2],[9,1],[10,0]]의 결과 :  [[-0.8531851]]

# learning_rate = 0.01
# loss : 2.2408278255170444e-06
# [[7,3],[8,2],[9,1],[10,0]]의 결과 :  [[-0.92925894]]

# learning_rate = 0.01
# y데이터 2개 구하기 (dance 2개로 변경)
# loss : 2.0759303879458457e-05
# [[7,3],[8,2],[9,1],[10,0]]의 결과 :  [[10.924551  -0.9182138]]


#y_data: [ 5  6  7  8  9 10] 일때 예측값 1개
# learning_rate = 0.01
# loss : 1.955170046130661e-05
# [[7,3],[8,2],[9,1],[10,0]]의 결과 :  [[10.803729]]

#y_data: [ 5  6  7  8  9 10] 일때 예측값 1개
# learning_rate = 0.05
# loss : 0.00034620449878275394
# [[7,3],[8,2],[9,1],[10,0]]의 결과 :  [[11.089691]]
