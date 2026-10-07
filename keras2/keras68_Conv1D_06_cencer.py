# 53-6 copy
# 회귀 → 숫자 예측 → R², MSE, RMSE
# 분류 → 종류 예측 → Accuracy, Precision, Recall, F1


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, MaxPool1D, GlobalAveragePooling1D, Flatten
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_breast_cancer 

#1. 데이터
datasets = load_breast_cancer()
x = datasets['data'] 
y = datasets['target']
print(x.shape, y.shape) #(569, 30) (569,)
print("y의 범주의 갯수 : ",np.unique(y, return_counts=True)) 

# 데이터셋 트레인,테스트 분리(7:3)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    random_state=908,
    test_size=0.3,
    shuffle=True,
    stratify=y, # (중요) y데이터를 스트레이트파이하란 이야기
    # stratify = '층을 이루게 하다', '계층화하다', '계층별로 나누다'
    )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)   

print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test))
print(x_train.shape, x_test.shape) #(398, 30) (171, 30)
print(y_train.shape, y_test.shape) #(398,) (171,)

# Dnn >>> Conv1D 바꾸기
x_train = x_train.reshape(-1, 6, 5)
x_test = x_test.reshape(-1, 6, 5)
print(x_train.shape, x_test.shape) #(398, 6, 5) (171, 6, 5)
print(y_train.shape, y_test.shape) #(398,) (171,)

#2. 모델구성
model = Sequential()

model.add(Conv1D(64, 3, input_shape=(6,5)))
model.add(Conv1D(64, 3, padding='same',activation='relu', ))
model.add(LSTM(32))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3.컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=learning_rate), 
              metrics=['accuracy'], 
              ) 


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    verbose=1,
    restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5, # learning_rate(러닝레이트) 비율 조절

)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=100,
          batch_size=32,
          validation_split=0.3, #트레인에서 검증부분 분리
          verbose=1,
          callbacks=[es, rlr],
          )
end_time = time.time()
print("걸린시간 :", round(end_time-start_time,2), "초")


#4. 성능, 평가
loss = model.evaluate(x_test, y_test)
print("loss : ", round(loss[0],4)) # loss=binary_crossentropy
print("accuracy : ", round(loss[1],4))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 반올림 처리

from sklearn.metrics import accuracy_score 
acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score ) 


### dnn >>> Conv1D : loss 개선
# loss :  0.0462
# accuracy :  0.9883
# acc_score :  0.9883040935672515

#  MinMaxScaler 적용 후  
# loss :  0.0681
# accuracy :  0.9883
# acc_score :  0.9883040935672515









