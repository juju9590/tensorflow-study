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
model.add(Dense(64, input_dim=1))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

############## 동결 ##############
model.trainable = False  # ******* 중요 **********  #훈련을 하지 않겠다.
# model.trainable = True

# model.summary()

#03. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose=1)

#04. 평가, 예측
y_pred = model.predict(x)
print(y_pred)

# model.trainable = True ㅁ
# Epoch 1000/1000
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6.7557e-05 
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
# [[1.0025697]
#  [2.0071957]
#  [3.0116074]
#  [4.0160184]
#  [5.02043  ]]

# model.trainable = False #초기의 랜덤값으로 동결되어 좋은값이 나올 수 없음
# Epoch 1000/1000
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 10.0707 
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step
# [[0.04317248]
#  [0.08634496]
#  [0.12951759]
#  [0.17268991]
#  [0.2158624 ]]
