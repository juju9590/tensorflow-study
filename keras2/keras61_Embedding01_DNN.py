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
]
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

# {'참': 1, '너무': 2, '재밌있다': 3, '최고에요': 4, '잘만든': 5, 
# '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, 
# '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, 
# '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, 
# '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '개똥이': 24, '바보': 25, 
# '말똥이': 26, '잘생겼다': 27, '길동이': 28, '또': 29, '구라친다': 30}

x = token.texts_to_sequences(docs)
print(x)

# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], 
# [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], 
# [24, 25], [26, 27], [28, 29, 30]]
# 데이터의 길이가 모두 다름 ==> 맞춰 줘야함 (앞이나 뒤를 패딩처리 한다.)

# [2, 3], 0    0    0
# [1, 4], 0    0    0
# [1, 5, 6],   0    0
# [7, 8, 9],   0    0 
# [10, 11, 12, 13, 14], 

#  0    0    0  [2, 3], 
#  0    0    0  [1, 4], 
#  0    0    [1, 5, 6], 
#  0    0    [7, 8, 9], 
# [10, 11, 12, 13, 14], 

########### 패딩 ###########
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x, 
                         padding='pre',   #post : 뒤를 0으로 채우다, pre : 앞을 0으로 채우다
                         maxlen = 5, # 디폴트 앞이 짤렸다
                        #  truncating='pre',# 디폴트
                        truncating='post', # 뒤가 짤린다
                         )
print(padded_x)
'''
[[ 0  0  0  2  3]
 [ 0  0  0  1  4]
 [ 0  0  1  5  6]
 [ 0  0  7  8  9]
 [10 11 12 13 14]
 [ 0  0  0  0 15]
 [ 0  0  0  0 16]
 [ 0  0  0 17 18]
 [ 0  0  0 19 20]
 [ 0  0  0  0 21]
 [ 0  0  0  2 22]
 [ 0  0  0  1 23]
 [ 0  0  0 24 25]
 [ 0  0  0 26 27]
 [ 0  0 28 29 30]]
'''
print(padded_x.shape) #(15, 5)
print(labels.shape) #(15,)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(padded_x, labels,
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
print(np.min(x_test), np.max(x_test)) # 0.0 11.0

# exit()

# x_predict = ['개똥이 잘생겼다']


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
          epochs=500, 
          batch_size=128, 
          verbose=1,
          validation_split=0.2,
          )

#4.평가, 예측
results = model.evaluate(x_test, y_test, verbose=1)

print('loss : ', round(results[0],2))
print('acc : ', round(results[1],2))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)


acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score )


# loss :  0.78
# acc :  0.6
# acc_score :  0.6

# loss :  0.58
# acc :  0.8
# acc_score :  0.8



