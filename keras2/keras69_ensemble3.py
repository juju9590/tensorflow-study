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
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
x3_datasets = np.array([range(100), range(301,401), range(77,177), range(33,133)]).transpose()

y1 = np.array(range(3001,3101)) # 온도 
y2 = np.array(range(13001,13101)) # 비트코인가격


x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516), range(249, 255)]).T
x3_pred = np.array([range(100,106), range(400,406), range(177,183), range(133,139)]).T

print(x1_pred.shape, x2_pred.shape, x3_pred.shape) #(6, 2) (6, 3) (6, 4)

# train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test , y2_train, y2_test= train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2,
    train_size=0.7, 
    random_state=567,
    )

print(x1_train.shape, x1_test.shape) #(70, 2) (30, 2)
print(x2_train.shape, x2_test.shape) #(70, 3) (30, 3)
print(x3_train.shape, x3_test.shape) #(70, 4) (30, 4)
print(y1_train.shape, y1_test.shape) #(70,) (30,)
print(y2_train.shape, y2_test.shape) #(70,) (30,)

#02-1 모델구성

input1 = Input(shape=(2,)) 
dense1 = Dense(64, activation='relu', name='han_1')(input1) 
dense2 = Dense(64, activation='relu', name='han_2')(dense1) 
dense3 = Dense(32, activation='relu', name='han_3')(dense2) 
output1 = Dense(16, activation='relu', name='han_4')(dense3) 


#02-2 모델구성

input21 = Input(shape=(3,)) 
dense21 = Dense(64, activation='relu', name='han_21')(input21) 
dense22 = Dense(64, activation='relu', name='han_22')(dense21) 
dense23 = Dense(32, activation='relu', name='han_23')(dense22) 
dense24 = Dense(32, activation='relu', name='han_24')(dense23) 
output21 = Dense(16, activation='relu', name='han_25')(dense24) 

#02-3 모델구성

input31 = Input(shape=(4,)) 
dense31 = Dense(32, activation='relu', name='han_31')(input31) 
dense32 = Dense(32, activation='relu', name='han_32')(dense31) 
output31 = Dense(16, activation='relu', name='han_33')(dense32) 


#02-4 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate

merge_1 = concatenate([output1, output21, output31 ], name="mg_1") #매서드
# 두 모델을 합쳐서 하나의 모델로 만들기 때문에 output만 연결한다
# merge_1 = Concatenate(name="mg_1")([output1, output21]) #클래스
# merge_2 = Dense(10, name='mg_2')(merge_1)
# merge_3 = Dense(5, name='mg_3')(merge_2)

#02-5 분기 1
# last_dense_1 = Dense(10, name='ld_1')(merge_3)
# last_dense_2 = Dense(10, name='ld_2')(last_dense_1)
last_output_1 = Dense(1, name='last1')(merge_1)

#02-6 분기 2
last_output_2 = Dense(1, name='last2')(merge_1)

model = Model(inputs=[input1, input21, input31], outputs=[last_output_1, last_output_2] )

model.summary()

#03. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit([x1_train, x2_train, x3_train ], [y1_train, y2_train],
          epochs=1500,
          batch_size=32,
          validation_split=0.3,
          )

#04. 평가, 예측
results = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test] )
print('loss_1 :', results[0])
print('loss_2 :', results[1])
print('loss_3 :', results[2])

y_pred = model.predict([x1_pred, x2_pred, x3_pred])
print("온도 :", y_pred[0], "가격 : ", y_pred[1])

# 결과
# last1_loss: 0.4893 - last2_loss: 5.7067 - loss: 6.1960
# loss_1 : 6.1960225105285645 => 전체
# loss_2 : 0.48932960629463196 => last1
# loss_3 : 5.706692695617676 => last2
# 온도 : 
# [[3098.3572]
#  [3101.4604]
#  [3104.5627]
#  [3107.7444]
#  [3110.9282]
#  [3114.1135]] 
# 가격 :  
# [[13095.416]
#  [13109.361]
#  [13123.306]
#  [13137.619]
#  [13151.945]
#  [13166.274]]

# last1_loss: 0.2124 - last2_loss: 2.7014 - loss: 2.9137
# loss_1 : 2.913727045059204
# loss_2 : 0.2123609036207199
# loss_3 : 2.7013661861419678
# 온도 : 
# [[3102.1362]
#  [3108.2312]
#  [3114.3254]
#  [3120.4202]
#  [3126.5142]
#  [3132.6082]] 
# 가격 :  
# [[13088.646]
#  [13104.391]
#  [13120.134]
#  [13135.878]
#  [13151.62 ]
#  [13167.364]]
