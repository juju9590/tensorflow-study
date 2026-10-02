# [실습] acc 0.6 이상

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
    num_words=5000,
#     maxlen=100,
    # test_split=0.2, # Unrecognized keyword arguments: {'test_split': 0.2}. 
)

print(x_train)
print(x_train.shape, y_train.shape) #(25000,) (25000,)
print(x_test.shape, y_test.shape) #(25000,) (25000,)
print(y_train) #[0 0 1 ... 0 1 1]
print(np.unique(y_train, return_counts=True))
# (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))

print(type(x_train)) #<class 'numpy.ndarray'>
print(type(x_train[0])) #<class 'list'>
print(len(x_train[0]), len(x_train[1])) # 218 189

print("최대길이 : ", max(len(i) for i in x_train)) #2494
print("최소길이 : ", min(len(i) for i in x_train)) #11
print("평균길이 : ", sum(map(len,x_train))/len(x_train)) #238.71364

# 전처리 (패드 시퀀스)
x_train = pad_sequences(x_train,
                         padding='pre',   
                         maxlen = 200, 
                         )
print(x_train.shape) # (25000, 200)

x_test = pad_sequences(x_test,
                         padding='pre',  
                         maxlen = 200, 
                         )
print(x_test.shape) #(25000, 200)

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
print(y_train.shape) #(25000, 2)
y_test = to_categorical(y_test)

#2. 모델구성
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, input_length=200)) #input_length=5 행무시 열우선
# model.add(Embedding(5000, 100)) #input_length=5 행무시 열우선
# model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(SimpleRNN(64, return_sequences=True))) 
model.add(LSTM(64, return_sequences=True,))
model.add(LSTM(32)) 
model.add(Dense(32))
model.add(Dense(2, activation='sigmoid')) # 이진분류 2개

model.summary()

#3. 컴파일, 훈련
start_time = time.time()

model.compile(
        loss="binary_crossentropy",  #이진분류 손실함수
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
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)


# 결과
# loss :  1.0
# acc :  0.85

# loss :  0.73
# acc :  0.86
# acc_score :  0.86276


