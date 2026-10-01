# 분류
# AI 모델은 분류(이진,다중 분류)와 회귀 모델 2가지만 있다.

# 33-6 카피

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_breast_cancer #데이터셋 불러오기(유방암관련)

#1. 데이터
datasets = load_breast_cancer()
# print(datasets.DESCR) 
# print(datasets.feature_names) 

# x = datasets.data
x = datasets['data'] 
y = datasets.target # ['target']

print(x.shape, y.shape) #(569, 30) (569,)
print('타입 : ', type(x)) 

print(y) #(y=범주) [0,1,0,0,...,1,1,1,0,1]

print("y의 범주 : ",np.unique(y)) # [0 1] 
print("y의 범주의 갯수 : ",np.unique(y, return_counts=True)) 

print(pd.DataFrame(y).value_counts()) 
# 1    357
# 0    212
print(pd.Series(y).value_counts())
# 1    357
# 0    212

# 데이터셋 트레인,테스트 분리(7:3)
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    random_state=908,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    stratify=y, # (중요) y데이터를 스트레이트파이하란 이야기
                                                    # stratify = '층을 이루게 하다', '계층화하다', '계층별로 나누다'
                                                    )

#  1) stratify=y 미설정 시 
# print(np.unique(y_train, return_counts=True))
# print(np.unique(y_test, return_counts=True))

# 중요 2) stratify=y 설정시
# print("==== y_train의 종류와 종류별 트레인(70%)로 분리 ====")
# print(np.unique(y_train, return_counts=True))
# (array([0, 1]), array([148, 250]))
# print("==== y_test의 종류와 종류별 테스트(30%)로 분리 ====")
# print(np.unique(y_test, return_counts=True))
# (array([0, 1]), array([ 64, 107])) ==> y를 기준으로 7:3 비율로 분리 해준다

print(x_train.shape, x_test.shape) # (398, 30) (171, 30)
print(y_train.shape, y_test.shape) # (398,) (171,)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train) 

x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) #-2.3737740158538205 18.736137494346455
print(np.min(x_test), np.max(x_test)) #-1.8878923766816142 5.996470172961524

########### X 데이터를 CNN에 넣기 위해 4차원으로 변환
x_train = x_train.reshape(-1,6,5,1)
x_test = x_test.reshape(-1,6,5,1)

print(x_train.shape, x_test.shape) #(398, 6, 5, 1) (171, 6, 5, 1)
print(y_train.shape, y_test.shape) #(398,) (171,)

# exit()

#2. 모델구성
model = Sequential()

model.add(Conv2D(64, (2,2), input_shape=(6, 5, 1))) 
model.add(Conv2D(32, (2,2), padding='same', activation='relu' )) 
model.add(MaxPooling2D())

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid')) #(필수)이진분류모델

model.summary()

# exit()

#3.컴파일, 훈련
model.compile(loss='binary_crossentropy', #(필수)이진분류모델
              optimizer='adam', # 아담이 그라디언트, 가중치 갱신 계산
              metrics=['accuracy'], 
              ) 


es = EarlyStopping(
    monitor="val_loss",
    mode='min',
    patience=20,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=100,
          batch_size=32,
          validation_split=0.3, #트레인에서 검증부분 분리
          verbose=1,
          callbacks=[es],
          )
end_time = time.time()
print("걸린시간 :", round(end_time-start_time,2), "초")

print("===================== 학습완료 ===========================")

#4. 성능, 평가
print("=========================================================")
loss = model.evaluate(x_test, y_test)
print("loss : ", round(loss[0],4)) # loss=binary_crossentropy
print("accuracy : ", round(loss[1],4))
print("=========================================================")



y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 반올림 처리

print(x_test.shape)
print(y_pred.shape)

from sklearn.metrics import accuracy_score 

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score ) 


# ############## RobustScaler 적용 후  ################### ==> loss 하향, 정확도 향상
# 걸린시간 : 2.86 초
# loss :  0.1388
# accuracy :  0.9649
# acc_score :  0.9649122807017544

############ 드롭아웃 ==> loss 향상
# loss :  0.0864
# accuracy :  0.9649
# acc_score :  0.9649122807017544

#### dnn >>>> cnn 1차
# 걸린시간 : 3.56 초
# loss :  0.0828
# accuracy :  0.9766
# acc_score :  0.9766081871345029

#### dnn >>>> cnn 1차 (cpu)
# loss :  0.0612
# accuracy :  0.9825
# acc_score :  0.9824561403508771
# 걸린시간 : 3.54 초

#### dnn >>>> cnn 2차 (cpu)
# 걸린시간 : 3.71 초
# loss :  0.1355
# accuracy :  0.9474
# acc_score :  0.9473684210526315


