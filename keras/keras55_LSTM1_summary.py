# 54-2 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
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

print(x.shape, y.shape) #(7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)  ### ([[[1],[2],[3]], ... [[[7],[8],[9]]]) 형식으로 바꿔줌
print(x.shape)  #(7, 3, 1)

#2. 모델구성
model = Sequential()

# model.add(SimpleRNN(10, input_shape=(3, 1))) # 
# model.add(LSTM(10, input_shape=(3, 1)))  
model.add(GRU(10,input_shape=(3,1)))
### RNN 계열의 끝판왕 LSTM
### RNN의 단점 , 데이터가 많을 수록 과거의 데이터를 까먹는다(가중치가 계속 연산되기 때문에 작아져 영향을 못 미친다.)

### 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
# input_shape=(3, 1) 3=timesteps, 1=feature
model.add(Dense(7, activation='relu')) 
model.add(Dense(1))

model.summary()

# 파라미터 개수 = units*feature + units*bias + units*units
        #   = units * (feature + bias + units)


##### SimpelRNN   [ model.add(SimpleRNN(10, input_shape=(3, 1))) ]  
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 10)                120       
                                                                 
#  dense (Dense)               (None, 7)                 77        
                                                                 
#  dense_1 (Dense)             (None, 1)                 8         
                                                                 
# =================================================================
# Total params: 205
# Trainable params: 205
# Non-trainable params: 0

##### LSTM model.add(LSTM(10, input_shape=(3, 1)))
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  lstm (LSTM)                 (None, 10)                480       
                                                                 
#  dense (Dense)               (None, 7)                 77        
                                                                 
#  dense_1 (Dense)             (None, 1)                 8         
                                                                 
# =================================================================
# Total params: 565
# Trainable params: 565
# Non-trainable params: 0
# _________________________________________________________________

# params = dim(W)+dim(V)+dim(U) = n*n + kn + nm

lstm = (10+10*10+10)*4
print(lstm) #480


# 파라미터 개수 = 4*(units*feature + units*units + units)
# LSTM이 RNN보다 연산량이 4배정도 많다. 성능은 RNN보다 뛰어나다
# 단점은 느리다