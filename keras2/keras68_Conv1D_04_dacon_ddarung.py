# 42-4 카피

import numpy as np # 수치 계산에 특화
import pandas as pd # sklearn 만큼 강력함
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv1D, LSTM, MaxPool1D, GlobalAveragePooling1D
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
path = "c:\study\_data\ddarung/" 

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(train_csv.shape) # (1459, 10) 
print(test_csv.shape) # (715, 9) 

#### 결측치 처리
train_csv = train_csv.dropna() # 결측치 있는 row 자체를 없애버림
# print(train_csv) # [1328 rows x 10 columns]

# train_csv를 x와 y로 분리
x = train_csv.drop(['count'], axis=1) # 열(컬럼) 삭제 | 행0, 열1
y = train_csv['count'] # pandas에서 컬럼만 빼는거

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=333)
print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
print(y_train.shape, y_test.shape) #(929,) (399,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test))

########### X 데이터를 CNN에 넣기 위해 4차원으로 변환
x_train = x_train.reshape(-1,9,1)
x_test = x_test.reshape(-1,9,1)

print(x_train.shape, x_test.shape) #(929, 9, 1) (399, 9, 1)
print(y_train.shape, y_test.shape) #(929,) (399,)

# 2. 모델 구성
model = Sequential()

model.add(Conv1D(64, 3, input_shape=(9, 1))) 
model.add(Conv1D(32, 3, padding='same', activation='relu' )) 
model.add(Conv1D(16, 2, padding='same', activation='relu' ))

model.add(Flatten())
# model.add(GlobalAveragePooling1D())

model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(1,))  

model.summary()

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")

import time
start_time = time.time()
hist = model.fit(x_train, y_train, 
                epochs=500, 
                batch_size=160,
                validation_split=0.2,
                )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)
print("r2 : ", r2_score(y_test, y_pred))
print("rmse : ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("걸린시간 :", round(end_time-start_time,2), "초")

### Conv1D (flatten)
# loss :  2062.96533203125
# r2 :  0.7135481218250596
# rmse :  45.41987975330343
# 걸린시간 : 17.08 초


### Conv1D (GAP)
# loss :  2256.96142578125
# r2 :  0.6866109378754238
# rmse :  47.50748724200515
# 걸린시간 : 16.09 초

###### dnn >>> > cnn 1차
# loss :  1809.003662109375
# r2 :  0.7488118498116965
# rmse :  42.53238308731484

###### dnn >>> > cnn 2차
# 걸린시간 : 46.48 초
# loss :  2380.10791015625 (0에 가까울수록 좋음)
# r2 :  0.6695114740855965 (1에 가까울수록 좋음)
# rmse :  48.7863510002996 (0에 가까울수록 좋음)

###### dnn >>> > cnn 2차
# loss :  2270.956298828125
# r2 :  0.6846676673123562
# rmse :  47.65455240756137
# 걸린시간 : 35.04 초