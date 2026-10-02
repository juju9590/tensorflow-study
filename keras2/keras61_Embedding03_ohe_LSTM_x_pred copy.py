# x_predict = ['개똥이 잘생겼다'] 예측하기

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
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
]
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

# 입력한 문장을 단어 사전으로 만들고 숫자 붙여주기
token = Tokenizer()
token.fit_on_texts(docs) 
print(token.word_index)
'''
{'참': 1, '너무': 2, '재밌있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '개똥이': 24, '바보': 25, '말똥이': 26, '잘생겼다': 27, '길동이': 28, '또': 29, '구라친다': 30}
'''
# 문장을 위에서 만든 숫자로 바꾸기
x = token.texts_to_sequences(docs)
# print(x)
'''
[[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]
'''
# 예측 문제
x_predict = ['개똥이 잘생겼다'] # 예측대상
x_predict = token.texts_to_sequences(x_predict)
print(x_predict) #[[24, 27]]

y_true = np.array([1]) # 예측정답 
print(y_true.shape) #(1,)

#패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x, 
                         padding='pre',   #post : 뒤, pre : 앞 => 0으로 채우다
                         maxlen = 5, # 디폴트 앞이 짤렸다
                        #  truncating='pre',# 디폴트
                        # truncating='post', # 뒤가 짤린다
                         )

padded_x = padded_x
print(padded_x)
'''
[[ 0  0  0  2  3]
 [ 0  0  0  1  4]
 ...
 [ 0  0  0 26 27]
 [ 0  0 28 29 30]]
'''
print(padded_x.shape) #(15, 5)
print(labels.shape) #(15,)


x_predict = pad_sequences(x_predict, 
                         padding='pre',   #post : 뒤, pre : 앞 => 0으로 채우다
                         maxlen = 5, # 디폴트 앞이 짤렸다
                        #  truncating='pre',# 디폴트
                        # truncating='post', # 뒤가 짤린다
                         )

print(x_predict) #[[ 0  0  0 24 27]]
print(x_predict.shape) #(1, 5)

from tensorflow.keras.utils import to_categorical

padded_x = to_categorical(padded_x)
print(padded_x)
'''[[1. 0. 0. ... 0. 0. 0.]
  [1. 0. 0. ... 0. 0. 0.]
  ...
 [0. 0. 0. ... 0. 1. 0.]
  [0. 0. 0. ... 0. 0. 1.]]]  
'''
print(padded_x.shape) #(15, 5, 31)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
                                padded_x, labels,
                                random_state=999,
                                test_size=0.2,
                                shuffle=True,
)

print(x_train.shape, x_test.shape) # 
print(y_train.shape, y_test.shape) # 


#2. 모델 구성
# 시그모이드, 원핫은 하지말고 

model = Sequential()
model.add(LSTM(64, input_shape=(5, 31)))  #행무시 열우선
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(
        loss="binary_crossentropy", 
        optimizer='adam',
        metrics = ['acc']
        )

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

y_pred = model.predict(x_predict)
y_pred = np.round(y_pred) # 반올림 처리

acc_score = accuracy_score(y_true, y_pred)  
print("acc_score : ", acc_score )



# loss :  3.65
# acc :  0.0

# loss :  0.78
# acc :  0.6
# acc_score :  0.6

# loss :  0.58
# acc :  0.8
# acc_score :  0.8



