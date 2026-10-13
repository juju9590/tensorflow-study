# trainable weights : 훈련가능한 가중치

import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

print(tf.__version__) #2.21.0

# 초기값을 고정시키는것 (통상적으로 작업할때 이렇게 고정해 놓고 작업한다)
random.seed(23) #파이썬 랜덤값 고정 (import random 호출필요)
np.random.seed(1333) # 여기서 작업하는 모든 넘파이는 랜덤표 333에 따라 작성한다. 넘파이 랜덤값 고정
tf.random.set_seed(569) # 텐서플로우 랜덤값 고정

#01. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])


#02. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
#  Total params: 17 (68.00 B)
#  Trainable params: 17 (68.00 B)
#  Non-trainable params: 0 (0.00 B)
 
print("======================= model.weights ==================================")
print(model.weights)
# [<Variable path=sequential/dense/kernel, shape=(1, 3), dtype=float32, # 입력 1, 출력 3
#  value=[[ 0.21477032  1.0670055  -0.5093527 ]]>, 
#  <Variable path=sequential/dense/bias, shape=(3,), dtype=float32, 
#  value=[0. 0. 0.]>,  # bias 초기값 = 0

#  <Variable path=sequential/dense_1/kernel, shape=(3, 2), dtype=float32, 
#  value=[[ 0.31022513 -0.65687835]
#  [-0.36185437  0.5525259 ]
#  [-0.11674267  0.24805963]]>, 
#  <Variable path=sequential/dense_1/bias, shape=(2,), dtype=float32, 
#  value=[0. 0.]>, 

#  <Variable path=sequential/dense_2/kernel, shape=(2, 1), dtype=float32, 
#  value=[[ 1.3845102 ]
#  [-0.85260814]]>, 
#  <Variable path=sequential/dense_2/bias, shape=(1,), dtype=float32, 
#  value=[0.]>]

print("======================== model.trainable_weights ==========================")
print(model.trainable_weights)

# [<Variable path=sequential/dense/kernel, shape=(1, 3), dtype=float32, 
#  value=[[ 0.21477032  1.0670055  -0.5093527 ]]>, 
#  <Variable path=sequential/dense/bias, shape=(3,), dtype=float32, 
#  value=[0. 0. 0.]>, 
 
#  <Variable path=sequential/dense_1/kernel, shape=(3, 2), dtype=float32, 
#  value=[[ 0.31022513 -0.65687835]
#  [-0.36185437  0.5525259 ]
#  [-0.11674267  0.24805963]]>, 
#  <Variable path=sequential/dense_1/bias, shape=(2,), dtype=float32, 
#  value=[0. 0.]>, 
 
#  <Variable path=sequential/dense_2/kernel, shape=(2, 1), dtype=float32, 
#  value=[[ 1.3845102 ]
#  [-0.85260814]]>, 
#  <Variable path=sequential/dense_2/bias, shape=(1,), dtype=float32, 
#  value=[0.]>]

print("======================== ===================== ==========================")
print(len(model.weights)) #6 
print(len(model.trainable_weights)) #6 


############## 동결 ##############
model.trainable = False  # ******* 중요 **********  #훈련을 하지 않겠다.
############## 동결 ##############
print(len(model.weights)) #6 
print(len(model.trainable_weights)) #0
print("======================== model.trainable_weights ==========================")
print(model.trainable_weights) #[]
 
print("======================== model.weights ==================================")
print(model.weights)
# [<Variable path=sequential/dense/kernel, shape=(1, 3), dtype=float32, value=[[ 0.21477032  1.0670055  -0.5093527 ]]>, <Variable path=sequential/dense/bias, shape=(3,), dtype=float32, value=[0. 0. 0.]>, <Variable path=sequential/dense_1/kernel, shape=(3, 2), dtype=float32, value=[[ 0.31022513 -0.65687835]
#  [-0.36185437  0.5525259 ]
#  [-0.11674267  0.24805963]]>, <Variable path=sequential/dense_1/bias, shape=(2,), dtype=float32, value=[0. 0.]>, <Variable path=sequential/dense_2/kernel, shape=(2, 1), dtype=float32, value=[[ 1.3845102 ]
#  [-0.85260814]]>, <Variable path=sequential/dense_2/bias, shape=(1,), dtype=float32, value=[0.]>]
# 가중치를 가지고 있지만, 훈련시키지 않는다.. ==> 이말은 이 가중치가 최종

model.summary()
#  Total params: 17 (68.00 B)
#  Trainable params: 0 (0.00 B)
#  Non-trainable params: 17 (68.00 B) # ************
# ㄴ 이 모델의 가중치는 갱신되지 않는다 

