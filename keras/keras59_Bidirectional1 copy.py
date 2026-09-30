# 54-1 copy
# Bidirectional : 양방향 모델

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional
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
           ])        

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) 

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)  

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))  # 행무시, 열우선 (3,1)이 7개 있다..로 해석
model.add(Bidirectional(SimpleRNN(10), input_shape=(3, 1)))
### simpleRNN을 양방향으로 2번 해라 
model.add(Dense(10)) 
model.add(Dense(10))

model.summary()

# Please initialize `Bidirectional` layer with a `tf.keras.layers.Layer` instance. Received: 10
# initialize : 초기화 
# Bidirectional 자체 모델이 아니다. 랩핑 모델을 넣어주어야 한다. 주로 RNN 안에 넣어준다
