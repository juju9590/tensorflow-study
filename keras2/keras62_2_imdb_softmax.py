# [실습] 목표 acc 0.6 이상
# [실습] 목표 acc 0.85 이상 (갱신)

# Embedding + LSTM/GRU
# Embedding + Bidrictional
# Embedding + Flatten + DNN

from tensorflow.keras.datasets import imdb

import time
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, SimpleRNN, SpatialDropout1D
from sklearn.metrics import accuracy_score

(x_train, y_train),(x_test, y_test) = imdb.load_data(
    num_words=1000, #단어사전의 갯수 = input_dim 
    maxlen=500,
    # test_split=0.2, # Unrecognized keyword arguments: {'test_split': 0.2}. 
)

print(x_train)
print(x_train.shape, y_train.shape) #(22882,) (22882,)
print(x_test.shape, y_test.shape) #(23065,) (23065,)
print(y_train) #[1 0 0 ... 0 1 0]
print(np.unique(y_train, return_counts=True))
# (array([0, 1], dtype=int64), array([11531, 11351], dtype=int64))

print(type(x_train)) #<class 'numpy.ndarray'>
print(type(x_train[0])) #<class 'list'>
print(len(x_train[0]), len(x_train[1])) # 218 189

print("최대길이 : ", max(len(i) for i in x_train)) #499
print("최소길이 : ", min(len(i) for i in x_train)) #11
print("평균길이 : ", sum(map(len,x_train))/len(x_train)) #196.66209247443405

# 전처리 (패드 시퀀스)
x_train = pad_sequences(x_train,
                         padding='pre',   
                        #  maxlen = 145, 
                         )
print(x_train.shape) # (22882, 499)

x_test = pad_sequences(x_test,
                         padding='pre',  
                        #  maxlen = 145, 
                         )
print(x_test.shape) # (23065, 499)

# y 원핫
y_train = to_categorical(y_train)
print(y_train)
'''
[[0. 1.]
 [1. 0.]
 [1. 0.]
 ...
 [1. 0.]
 [0. 1.]
 [1. 0.]]
'''
print(y_train.shape) #(22882, 2)
y_test = to_categorical(y_test)

#2. 모델구성
model = Sequential()
# model.add(Embedding(input_dim=1000, output_dim=150, input_length=145)) #input_length=5 행무시 열우선
model.add(Embedding(1000, 100)) #input_length=5 행무시 열우선
# model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(SimpleRNN(64, return_sequences=True))) 
# model.add(LSTM(64, return_sequences=True,))
model.add(LSTM(32)) 
# return_sequences=False(디폴트) 로 3차원((batch, timesteps, embedding_dim)에서  2차원(batch, 46)으로 바꿔줘야함
# ㄴ 이건 timestep 축을 없애는 역할
# Dense(64)로 연결되어서가 아니라, 마지막에 model.add(Dense(46, activation='softmax'))인
# 다중분류로 출력해야 하기 때문에 2차원(batch, 46)으로 출력되어야 한다.
model.add(Dense(32))
model.add(Dense(2, activation='softmax')) # 다중분류 2개

model.summary()

#3. 컴파일, 훈련
start_time = time.time()
model.compile(
        loss="categorical_crossentropy",  #다중분류 손실함수
        optimizer='adam',
        metrics = ['acc']
        )

model.fit(x_train, y_train,
        epochs=20, 
        batch_size=128, 
        verbose=1,
        )

end_time = time.time()
print("걸린시간 :", round(end_time-start_time,3),"초")


#4. 평가, 예측
results = model.evaluate(x_test, y_test, verbose=1)

print('loss : ', round(results[0],2))
print('acc : ', round(results[1],2))

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)


# 결과 (목표 acc 0.6이상 달성)

# 조건 model.add(Embedding(1000, 50)), maxlen=100,epochs=20,
# loss :  1.47
# acc :  0.8
# acc_score :  0.7958150523118461

# 조건 model.add(Embedding(1000, 100)), maxlen=499,epochs=20,
# 훈련 24분정도 걸림
# loss :  0.44
# acc :  0.86
# acc_score :  0.8593106438326469

