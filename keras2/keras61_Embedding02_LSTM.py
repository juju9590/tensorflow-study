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
print(padded_x)
print(padded_x.shape) #(15, 5) ==> 15,5,1
print(labels.shape) #(15,)

padded_x = padded_x.reshape(-1,5,1)
print(padded_x.shape) #(15, 5, 1)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(padded_x, labels,
                                                    random_state=999,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    )

print(x_train.shape, x_test.shape) # (10, 5, 1) (5, 5, 1)
print(y_train.shape, y_test.shape) # (10,) (5,)

# 스케일링
# 스케일링을 위해 2차원으로 변경
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
print(x_train.shape, x_test.shape) # (50, 1) (25, 1)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) # 0.0 1.0
print(np.min(x_test), np.max(x_test)) # 0.0 0.7333333333333333

# RNN 계열 모델 넣기위해 3차원으로 다시 변경
x_train = x_train.reshape(-1,5,1)
x_test = x_test.reshape(-1,5,1)
print(x_train.shape, x_test.shape) #(10, 5, 1) (5, 5, 1)

#2. 모델 구성
# 시그모이드, 원핫은 하지말고 

model = Sequential()
model.add(LSTM(64, input_shape=(5, 1)))  #행무시 열우선
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", 
              optimizer='adam',
              metrics = ['acc'])

model.fit(x_train, y_train,
          epochs=500, 
          batch_size=132, 
          verbose=1,
          validation_split=0.2,
          )

#4.평가, 예측
results = model.evaluate(x_test, y_test, verbose=1)

print('loss : ', round(results[0],2))
print('acc : ', round(results[1],2))

print(x_test.shape, y_test.shape) #(5, 5, 1) (5,)

y_pred = model.predict(x_test)
print(y_pred.shape) #(5, 1)

y_pred = np.round(y_pred).reshape(-1)
print(y_pred.shape) #(5,)

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score )



# loss :  3.65
# acc :  0.0

# loss :  0.78
# acc :  0.6
# acc_score :  0.6

# loss :  0.58
# acc :  0.8
# acc_score :  0.8



