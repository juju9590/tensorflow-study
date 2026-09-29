# [실습]
# loss는 0.1 이하
# 결과는 [101,102,103,104,105,106]의 근사치가 나오면 됨

import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau


a = np.array(range(1,101))
x_predict = np.array(range(96,106)) #101~106까지 찾자

size = 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size) 
print(bbb.shape) # (95, 6) => batch=94, timestep=7, feature=1 (1열)
# print(bbb) 

x = bbb[:, :-1]
# print(x, x.shape) #(94, 6)

y = bbb[:, -1]
# print(y, y.shape) #(94,)  

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(64, input_shape=(5, 1))) 
# model.add(LSTM(64, input_shape=(5, 1)))
# model.add(GRU(64, input_shape=(5, 1))) 
# model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))

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
    patience=40,
    verbose=1,
    restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=30,
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

####
x_predict = split_x(x_predict, size) 
# print(x_predict) 
'''
[[ 96  97  98  99 100 101]
 [ 97  98  99 100 101 102]
 [ 98  99 100 101 102 103]
 [ 99 100 101 102 103 104]
 [100 101 102 103 104 105]]
'''
# print(x_predict.shape) #(5, 6)

x_pred = x_predict.reshape(-1,5,1)
# print(x_pred.shape) #(6, 5, 1)

y_pred = model.predict(x_pred)

print('[101,102,103,104,105,106]의 결과 : ', np.round(y_pred,2))


# loss : 0.002619214588776231
# [101,102,103,104,105,106]의 결과 :  
# [[100.54]
#  [100.68]
#  [101.42]
#  [102.26]
#  [103.08]
#  [104.1 ]]


