# [실습] acc = 0.67 이상
# [실습] acc = 0.75 이상 상향시키기


from tensorflow.keras.datasets import reuters  #로이터통신 신문기사
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, SimpleRNN, SpatialDropout1D
from sklearn.metrics import accuracy_score

(x_train, y_train),(x_test, y_test) = reuters.load_data(
    num_words=1000, #단어사전의 갯수, 빈도수가 높은 단어순으로 1000개 뽑겠다
    # maxlen=100, # 단어 갯수 최대 길이 제한 
    test_split=0.2,
)
print(x_train)
print(x_train.shape, y_train.shape) #(8982,) (8982,)
print(x_test.shape, y_test.shape) #(2246,) (2246,)
print(y_train) #[ 3  4  3 ... 25  3 25]
print(np.unique(y_train, return_counts=True))
# 판다스의 벨류카운트랑 동일
# # (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], dtype=int64
# 다중분류, 46개, 손실함수 categorical_crossentropy,원핫인코딩
# 데이터 리스트 -> 넘파이로 바꿔주기 np,array ==> 패드 시퀀스 하면 넘파이로 변경된다 

print(type(x_train)) #<class 'numpy.ndarray'>
print(type(x_train[0])) #<class 'list'>
print(len(x_train[0]), len(x_train[1])) # 87 56

print("최대길이 : ", max(len(i) for i in x_train)) # 2376
print("최소길이 : ", min(len(i) for i in x_train)) # 13
print("평균길이 : ", sum(map(len,x_train))/len(x_train)) # 145.5398574927633

# 전처리 (패드 시퀀스)
x_train = pad_sequences(x_train,
                         padding='pre',   #post : 뒤를 0으로 채우다, pre : 앞을 0으로 채우다
                         maxlen = 145, # 디폴트 앞이 짤렸다
                        #  truncating='pre',# 디폴트
                        # truncating='post', # 뒤가 짤린다
                         )
print(x_train.shape) #(8982, 100)

x_test = pad_sequences(x_test,
                         padding='pre',   #post : 뒤를 0으로 채우다, pre : 앞을 0으로 채우다
                         maxlen = 145, # 디폴트 앞이 짤렸다
                        #  truncating='pre',# 디폴트
                        # truncating='post', # 뒤가 짤린다
                         )
print(x_test.shape) #(2246, 100)

# y 원핫
y_train = to_categorical(y_train)
print(y_train)
'''
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
'''
print(y_train.shape) #(8982, 46)
y_test = to_categorical(y_test)

#2. 모델구성
model = Sequential()
# model.add(Embedding(input_dim=1000, output_dim=150, input_length=145)) #input_length=5 행무시 열우선
model.add(Embedding(1000, 50)) #input_length=5 행무시 열우선
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(SimpleRNN(64, return_sequences=True))) 
model.add(LSTM(64, return_sequences=True,))
model.add(LSTM(32)) 
# return_sequences=False(디폴트) 로 3차원((batch, timesteps, embedding_dim)에서  2차원(batch, 46)으로 바꿔줘야함
# ㄴ 이건 timestep 축을 없애는 역할
# Dense(64)로 연결되어서가 아니라, 마지막에 model.add(Dense(46, activation='softmax'))인
# 다중분류로 출력해야 하기 때문에 2차원(batch, 46)으로 출력되어야 한다.
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(46, activation='softmax')) # 다중분류 46개

model.summary()

#3. 컴파일, 훈련
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

#4. 평가, 예측
results = model.evaluate(x_test, y_test, verbose=1)

print('loss : ', round(results[0],2))
print('acc : ', round(results[1],2))

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)


# 결과 (목표 acc 0.67 달성)
# loss :  2.35
# acc :  0.69
# acc_score :  0.6883348174532502

# 결과 : output_dim = 150 -> 200 : 소폭 상향
# loss :  1.29
# acc :  0.7
# acc_score :  0.699020480854853
