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
print(padded_x.shape) #(15, 5)
# (15, 5) = 15개의 문장, 최대 긴 단어의 수 5

#2. 모델
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

model = Sequential()

# 임베딩은 2차원 입력 => "3차원"으로 출력
# 그래서,,, 임베딩 다음엔 보통 RNN계열 모델이 붙는다

############# 임베딩 1 ############
model.add(Embedding(input_dim=30, output_dim=10 ,input_length=5)) #input_length=5 행무시 열우선
model.add(SimpleRNN(10))
model.add(Dense(1))
# input_dim : 단어사전의 갯수
# output_dim : 차원
'''
input_dim=30          output_dim=10 : 백터 데이터화 
[0]                => [0.1, 0.001, ..., 0.01]     #(10,) = 벡터 10개 = 하나의 값
[1]                => [0.1, 0.001, ..., 0.03]     #(10,) = 벡터 10개 = 하나의 값
[2]
...                            ...
[28]
[29]                => [0.1, 0.001, ..., 0.03]    #(10,) = 벡터 10개 = 하나의 값
                        ㄴ    ㄴ 디멘션    ㄴ 디멘션

                        벡터의 모임에서 한 열씩 디멘션 이라고 하고,, 벡터 자체는 하나의 값이다.
'''
## 백터DB

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  embedding (Embedding)       (None, 5, 10)             300        input_dim * output_dim = 300 (메모리공간)
                                                                 
#  simple_rnn (SimpleRNN)      (None, 10)                210       
# =================================================================

#input_dim: 단어 사전의 크기를 말하며 
# 총 max_features(15000)개의 단어 종류가 있다는 의미입니다. 
# 이 값은 앞서 reuters.load_data() 함수의 num_words 인자값과 동일해야 합니다.

#input_length: 단어의 수 즉 문장의 길이를 나타냅니다. 
# 임베딩 레이어의 출력 크기는 샘플 수 

# * output_dim * input_lenth가 됩니다. 
# 임베딩 레이어 다음에 Flatten 레이어가 온다면 반드시 input_lenth를 지정해야 합니다. 
# 플래튼 레이어인 경우 입력 크기가 알아야 이를 1차원으로 만들어서 
# Dense 레이어에 전달할 수 있기 때문입니다.


########## 임베딩 2 ##########
model.add(Embedding(input_dim=30, output_dim=10)) #input_length=5 는 명시 안해도 알아서 맞춰준다
model.add(SimpleRNN(10))
model.add(Dense(1))

########## 임베딩 3 ##########
# model.add(Embedding(30, 10))  #input_dim=30, output_dim=10 생략해서 쓸 수 있다.
# model.add(Embedding(30, 10, 5))  #input_length=5는 생략 안됨, ValueError: Could not interpret initializer identifier: 5
model.add(Embedding(30, 10, input_length=5)) # 이렇게 사용할 수 있다
model.add(SimpleRNN(10))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(
        loss="binary_crossentropy", 
        optimizer='adam',
        metrics = ['acc']
        )

model.fit(padded_x, labels,
        epochs=5, 
        batch_size=32, 
        verbose=1,
        )
