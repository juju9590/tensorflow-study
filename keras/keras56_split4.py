# 데이터를 reshape 한 후 , split_x 함수로 시계열데이터로 변환
# (N, 10, 1) -> (N, 5, 2)


import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau


x_predict = np.array(range(96,106))
a = np.array(range(1,101))


print(a.shape) #(100,)
a = a.reshape(-1,2)
# print(a)
print(a.shape) #(50, 2)

size = 6
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size) 
print(bbb.shape) #(45, 6, 2)

x = bbb[: , :-1, :]
print(x.shape)       #(45, 5, 2)
y = bbb[: , -1, 1]
print(y.shape)       #(45,)

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(32, input_shape=(5, 2)))  # 행무시, 열우선
# model.add(LSTM(64, input_shape=(5, 2))) 
# model.add(GRU(64, input_shape=(5, 2))) 
model.add(Dense(32))
model.add(Dense(16))

model.add(Dense(1))

model.summary()

#3. 컴파일

from tensorflow.keras.optimizers import Adam

# learning_rate = 0.1
# learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.00005
# learning_rate = 0.05
learning_rate = 0.005

model.compile(
    loss="mse", 
    optimizer=Adam(learning_rate=learning_rate),
)

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
    factor=0.1, # learning_rate(러닝레이트) 비율 조절
)

model.fit(x,y, 
    epochs=500, 
    batch_size=128,
    verbose=1,
    callbacks = [es, rlr ],
    )

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss :', results)

x_predict = np.array(range(96,106))
x_pred = x_predict.reshape(-1,5,2)
print(x_pred)

y_pred = model.predict(x_pred)

print('np.array(range(96,106))의 결과 : ', y_pred)


# x_pred
#  [[[ 96  97]
#   [ 98  99]
#   [100 101]
#   [102 103]
#   [104 105]]]

# loss : 0.07513189315795898
# np.array(range(96,106))의 결과 :  [[104.34409]]

# loss : 0.20606423914432526
# np.array(range(96,106))의 결과 :  [[108.00097]]

# loss : 0.003060681279748678
# np.array(range(96,106))의 결과 :  [[106.27014]]

# loss : 0.0036036265082657337
# np.array(range(96,106))의 결과 :  [[106.5202]]













