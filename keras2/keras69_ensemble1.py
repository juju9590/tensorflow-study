# 예측값까지 완성

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, mean_squared_error


#01. 데이터
x1_datasets = np.array([range(100), range(301,401)]).T
                    # 삼성 종가        하이닉스 종가
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
                    # 원유가             환율             금시세
# print(x1_datasets.shape) #(100, 2)
# print(x2_datasets.shape) #(100, 3)
y = np.array(range(3001,3101)) #Mars의 화씨 온도
# print(y.shape) #(100, )

x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516), range(249, 255)]).T

print(x1_pred.shape, x2_pred.shape) #(6, 2) (6, 3)

# train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y, 
    train_size=0.8, 
    random_state=777,
    )

print(x1_train.shape, x2_train.shape, x1_test.shape, x2_test.shape) #(80, 2) (80, 3) (20, 2) (20, 3)
print(y_train.shape, y_test.shape) #(80,) (20,)


#02-1 모델구성

input1 = Input(shape=(2,)) 
dense1 = Dense(10, activation='relu', name='han_1')(input1) 
dense2 = Dense(20, activation='relu', name='han_2')(dense1) 
dense3 = Dense(30, activation='relu', name='han_3')(dense2) 
output1 = Dense(40, activation='relu', name='han_4')(dense3) 

# model_1 = Model(inputs=input1, outputs=output1)

#02-2 모델구성

input21 = Input(shape=(3,)) 
dense21 = Dense(50, activation='relu', name='han_21')(input21) 
dense22 = Dense(40, activation='relu', name='han_22')(dense21) 
dense23 = Dense(30, activation='relu', name='han_23')(dense22) 
dense24 = Dense(20, activation='relu', name='han_24')(dense23) 
output21 = Dense(10, activation='relu', name='han_25')(dense24) 

# model_2 = Model(inputs=input21, outputs=output21)

#02-3 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate

merge_1 = concatenate([output1, output21], name="mg_1") #매서드
# 두 모델을 합쳐서 하나의 모델로 만들기 때문에 output만 연결한다
# merge_1 = Concatenate(name="mg_1")([output1, output21]) #클래스
merge_2 = Dense(10, name='mg_2')(merge_1)
merge_3 = Dense(5, name='mg_3')(merge_2)
last_output = Dense(1, name='last')(merge_3)

model = Model(inputs=[input1, input21], outputs=last_output)

model.summary()

#03. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit([x1_train, x2_train], y_train,
          epochs=500,
          batch_size=16,
          validation_split=0.2,
          )

#04. 평가, 예측
results = model.evaluate([x1_test, x2_test], y_test)
print('loss :', results)

y_pred = model.predict([x1_pred, x2_pred])

print(y_pred.shape)
print(y_pred)

# 결과 
# loss : 2.127610921859741
# (6, 1)
# [[3107.643 ]
#  [3114.9092]
#  [3122.1758]
#  [3129.4421]
#  [3136.7087]
#  [3143.975 ]]


















