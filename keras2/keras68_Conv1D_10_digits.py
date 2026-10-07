# 42-10 카피
# acc = 1.0

from sklearn.datasets import load_digits

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D,GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                    test_size=0.2,
                                    random_state=333,
                                    shuffle=True,
                                    stratify=y,
                                    )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = RobustScaler() 
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)    
print(np.min(x_train), np.max(x_train)) # -2.6 16.0
print(np.min(x_test), np.max(x_test)) # -2.6 16.0

# print(x_train.shape, x_test.shape) # (1437, 64) (360, 64)
# print(y_train.shape, y_test.shape) # (1437,) (360,)

# Conv2D >>> Conv1D
x_train = x_train.reshape(-1,64,1)
x_test = x_test.reshape(-1,64,1)
# print(x_train.shape, x_test.shape) # (1437, 64, 1) (360, 64, 1)
# print(y_train.shape, y_test.shape) # (1437,) (360,)

#2. 모델구성
model = Sequential()

model.add(Conv1D(64, 2, input_shape=(64, 1))) 
model.add(Conv1D(64, 2 , activation='relu' )) 
model.add(Conv1D(64, 2, padding='same', activation='relu' )) 
# model.add(MaxPooling1D())

# model.add(GlobalAveragePooling1D())
model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer='adam', 
    metrics=['acc'],
)

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=1000,
          batch_size=64,
          validation_split=0.2,
          callbacks=[es],
          )
end_time = time.time()

print("걸린시간 :", round(end_time-start_time,2), "초")

#4. 예측, 평가
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", result[1])

y_pred = np.argmax(model.predict(x_test), axis=1)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", acc_score)



####### dnn >>> cnn (3차)
# 걸린시간 : 47.97 초
# loss : 0.053280770778656006
# acc : 0.9888888597488403
# acc_score : 0.9888888888888889

####### cnn2D >> Conv1D
# loss : 0.261295348405838
# acc : 0.9361110925674438
# acc_score : 0.9361111111111111
# 걸린시간 : 4.61 초

####### cnn2D >> Conv1D
# loss : 0.22027607262134552
# acc : 0.9583333134651184
# acc_score : 0.9583333333333334
# 걸린시간 : 4.19 초



