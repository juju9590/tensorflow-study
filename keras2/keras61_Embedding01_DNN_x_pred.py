# [실습] acc : 1
# x_predict = ['개똥이 잘생겼다']의 acc 구하기


import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

#1. 데이터
docs =[
    '너무 재밌있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다','한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요','너무 재미없다', '참 재밌네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다',
    '개똥이 잘생겼다',
]
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

########### 패딩 ###########
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x, 
                         padding='pre',   #post : 뒤를 0으로 채우다, pre : 앞을 0으로 채우다
                         maxlen = 5, # 디폴트 앞이 짤렸다
                        #  truncating='pre',# 디폴트
                        truncating='post', # 뒤가 짤린다
                         )
print(padded_x[:-1])
print(padded_x[:-1].shape) #(15, 5)
print(labels[:-1].shape) #(15,)

x_predict = padded_x[-1]
print(x_predict)  # [0 0 0 3 4]
print(x_predict.shape) #(5,)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(padded_x[:-1], labels[:-1],
                                                    random_state=999,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    )

print(x_train.shape, x_test.shape) # (10, 5) (5, 5)
print(y_train.shape, y_test.shape) # (10,) (5,)

# 스케일링
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) # 0.0 1.0
print(np.min(x_test), np.max(x_test)) # 0.0 13.0

print(x_predict.shape) #(5,)
x_predict = x_predict.reshape(1,5)
print(x_predict.shape) #(1, 5)
x_predict = scaler.transform(x_predict)
print(np.min(x_predict), np.max(x_predict)) #0.0 0.10344827586206896

#2. 모델 구성
# 시그모이드, 원핫은 하지말고 

model = Sequential()

model.add(Dense(64, input_shape=(5,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", 
              optimizer='adam',
              metrics = ['acc'])

model.fit(x_train, y_train,
          epochs=5, 
          batch_size=32, 
          verbose=1,
          validation_split=0.2,
          )

#4.평가, 예측
results = model.evaluate(x_test, y_test, verbose=1)

print('loss : ', round(results[0],2))
print('acc : ', round(results[1],2))

print(x_predict.shape) #(1, 5)
y_pred = model.predict(x_predict)
print(y_pred.shape) #(1, 1)

y_pred = np.round(y_pred).reshape(-1)

y_col = np.array([1])
print(y_col.shape) #(1,)

acc_score = accuracy_score(y_col, y_pred)  
print("acc_score : ", acc_score )

# loss :  1.11
# acc :  0.6
# (1, 5)
# (1, 1)
# (1,)
# acc_score :  1.0


# x_predict = ['개똥이 잘생겼다']의 acc 구하기
# loss :  0.34
# acc :  1.0
# (5, 1)
# (1, 5)
# acc_score :  1.0



