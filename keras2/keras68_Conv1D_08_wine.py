# 52-8 카피
## acc = 0.95 이상

from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, MaxPool1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(178, 13) (178,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=999,
    shuffle=True,
    stratify=y,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test))

print(x_train.shape, x_test.shape) # (142, 13) (36, 13)
print(y_train.shape, y_test.shape) # (142,) (36,)

### dnn >>> Conv1D
x_train = x_train.reshape(-1,13,1)
x_test = x_test.reshape(-1,13,1)
print(x_train.shape, x_test.shape) # (142, 13, 1) (36, 13, 1)
print(y_train.shape, y_test.shape) # (142,) (36,)

print(np.unique(y_train)) #[0 1 2]

# 2. 모델구성
model = Sequential()

model.add(Conv1D(64, 2, padding='same', activation='relu', input_shape=(13,1)))
model.add(Conv1D(64, 2, padding='same', activation='relu',))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax')) 

model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=Adam(learning_rate=learning_rate), 
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=500, 
          batch_size=64,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
          )

end_time = time.time()
print("걸린시간 :", round((end_time-start_time),2),"초")

# 4. 평가, 예측

result = model.evaluate(x_test, y_test) 
print("loss :", result[0])
print("acc :", result[1])

y_pred = np.argmax(model.predict(x_test), axis=1)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score : ", acc_score)


# ### dnn >>> conv1D (성능향상)
# loss : 0.011915832757949829
# acc : 1.0
# acc_score :  1.0

####### StandardScaler 적용 후 
# 걸린시간 : 30.17 초
# loss : 0.1700826734304428
# acc : 0.9722222089767456
# acc_score :  0.9722222222222222

# learning_rate = 0.01
# 걸린시간 : 16.55 초
# loss : 0.0782240703701973
# acc : 0.9722222089767456
# acc_score :  0.9722222222222222



