# RNN은 타임시리즈라고 불림

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

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
model.add(SimpleRNN(10, input_shape=(3, 1))) 
### 3차원으로 들어가서 1차원 또는 2차원으로 나옴 -> 바로 Dense와 연결가능

# RNN은 3차원 (batch, time steps, featuer)
# batch : 
# time steps : 길게 하느냐, 짧게 하느냐
# featuer