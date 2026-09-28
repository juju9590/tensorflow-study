# 54-1 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
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
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))  # 행무시, 열우선 (3,1)이 7개 있다..로 해석

model.add(SimpleRNN(10, input_shape=(3, 1))) 

### 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
# input_shape=(3, 1) 3=timesteps, 1=feature
model.add(Dense(7, activation='relu')) 
model.add(Dense(1))

model.summary()

# 파라미터 개수 = units*feature + units*bias + units*units
        #   = units * (feature + bias + units)

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 5)                 35        w, b 의 갯수 
#  dense (Dense)               (None, 7)                 42         5*7+7
#  dense_1 (Dense)             (None, 1)                 8          7*1+1
# =================================================================
# Total params: 85
# Trainable params: 85
# Non-trainable params: 0

# 입력 1 * 5(뉴런) + 5(뉴런)*5(은닉-순환) + 5

# 5+25+5

# # 타임스텝의 길이 100, 어휘사전 크기 300
# model.add(keras.layers.SimpleRNN(8, input_shape=(100, 300)))
# model.add(keras.layers.Dense(1, activation='sigmoid'))
#      - RNN 클래스 1개 추가, Dense 층 1개 추가
#          - RNN 클래스 가중치 : 300개의 입력, 8개의 뉴런, 순환되는 은닉상태는 출력과 동일하다 8개가 있다. 8개의 절편이 있다.
#             - 300(입력) x 8(뉴런) [완전연결] + 8(뉴런) x 8(은닉) [완전연결 순환] + 8(절편)
#          - Dense 층의 가중치 : 8개의 출력에 1개의 가중치가 곱해지고 1개의 절편이 있기 때문에 9개가 된다.
#             - 8(입력)  x 1(뉴런) + 1(절편)
# 출처: https://devspoon.tistory.com/196 [devspoon 오픈소스 개발자 번뇌 일지:티스토리]