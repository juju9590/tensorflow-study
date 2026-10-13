# trainable weights : 훈련가능한 가중치

import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

random.seed(23) 
np.random.seed(1333) 
tf.random.set_seed(569)

#01. 데이터
x = np.array([1,])
y = np.array([1,])

#02. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

############## 동결 ##############
# model.trainable = False  
model.trainable = True

# model.summary()



#03. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose=1)

#04. 평가, 예측
y_pred = model.predict(x)
print(y_pred)


print("==============================================")
print(model.weights)
print("==============================================")

# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step
# [[1.000001]]
# ===> 최적화된 가중치...
# [<Variable path=sequential/dense/kernel, shape=(1, 3), dtype=float32, 
# value=[[ 0.34775293  0.9530236  -0.6534927 ]]>, 
# <Variable path=sequential/dense/bias, shape=(3,), dtype=float32, 
# value=[ 0.13298257 -0.11398203 -0.14413968]>, 

# <Variable path=sequential/dense_1/kernel, shape=(3, 2), dtype=float32, 
# value=[[ 0.46169966 -0.8085058 ]
#  [-0.2458116   0.43664923]
#  [-0.25515762  0.386497  ]]>, 
# <Variable path=sequential/dense_1/bias, shape=(2,), dtype=float32, 
# value=[ 0.12346341 -0.12335327]>, 
# <Variable path=sequential/dense_2/kernel, shape=(2, 1), dtype=float32, 
# value=[[ 1.3979547]
#  [-0.8743269]]>, 
# <Variable path=sequential/dense_2/bias, shape=(1,), dtype=float32, 
# value=[0.12403195]>]