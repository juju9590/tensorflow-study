from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.preprocessing import OneHotEncoder # 전처리

text1 = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '개똥이는 기관사를 좋아한다. 말똥이는 잘생겼다. 길동이는 마구 마구 더 잘생겼다'

token = Tokenizer() # 인스턴스(객체) = 클래스를 정의했다 = 인스턴스 생성

token.fit_on_texts([text1,text2 ])
print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '잘생겼다': 4, '나는': 5, '지금': 6, 
# '맛있는': 7, '김밥을': 8, '엄청': 9, '먹었다': 10, '개똥이는': 11, 
# '기관사를': 12, '좋아한다': 13, '말똥이는': 14, '길동이는': 15, '더': 16}

x = token.texts_to_sequences([text1,text2])
print(x)
# [[5, 6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 1, 10], [11, 12, 13, 14, 4, 15, 1, 1, 16, 4]]

x1 =np.array(x[0])
x2 =np.array(x[1])
print("x1:", x1,"x2:",x2)

#힌트
x = np.concatenate((x1, x2))
print(x)
print(x.shape)

# 원핫인코딩
# 3. scikit learn → OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
x = np.array(x)              #리스트를 넘파이로 변환
x = x.reshape(-1,1)          #원핫인코딩을위한 reshape 실행
# print(x)
# print(x.shape)

x = ohe.fit_transform(x)
# print(x)
# print(x.shape)


