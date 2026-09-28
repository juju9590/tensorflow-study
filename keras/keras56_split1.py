# 시계열 timesteps 따라 데이터 나누기

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

a = np.array(range(1,11))
size = 5                # timestep 사이즈 

print(a.shape)           # (10,) 벡터, feature=1열짜리

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
'''
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]
 '''
print(bbb.shape) #(6, 5) => batch=6, timestep=5, feature=1 (1열)
print(len(a)) #10

# # subset
#         #인덱스  0 1 2 3 4             ==> x는 4열(특성은 4개)
# print(a[0:5]) #[1 2 3 4 5]
# print(a[1:6]) #[2 3 4 5 6] 
# print(a[2:7]) #[3 4 5 6 7] 
# print(a[3:8]) #[4 5 6 7 8] 
# print(a[4:9]) #[5 6 7 8 9] 
# print(a[5:10]) #[ 6  7  8  9 10] 

# x, y 데이터 분리 (리스트 Split)
# bbb[행, 열]
# 1)
# x_data = bbb[:, :4] # 모든 행, 인덱스 0부터 3까지 (4열 선택)
# y_data = bbb[:, 4]  # 모든 행, 인데스 4열 선택

# print("x_data", x_data)
# print("y_data", y_data)

# 2)
# print(bbb.shape[1]) #5
x_data = bbb[:, :(bbb.shape[1]-1)] # 모든 행, 인덱스 0부터 3까지 (4열 선택)
y_data = bbb[:, (bbb.shape[1]-1)]  # 모든 행, 인데스 4열 선택

# print("x_data", x_data)
# '''
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]]

# '''
# print("y_data", y_data) # [ 5  6  7  8  9 10]

# exit()


#2. 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(4, 1))) # x 특성이 4개
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))

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
learning_rate = 0.009

model.compile(loss="mse", 
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

model.fit(x_data,y_data, 
          epochs=500, 
          batch_size=128,
          verbose=1,
          callbacks = [es, rlr ],
          )

#4. 평가, 예측
results = model.evaluate(x_data, y_data)
print('loss :', results)


# [7, 8, 9, 10]은 1차원 배열로 RNN의 3차원 형식으로 변경 필요
x_predict =np.array([7,8,9,10])
x_pred = x_predict.reshape(1,4,1)

y_pred = model.predict(x_pred)

print('[7,8,9,10]의 결과 : ', y_pred)

# 결과 (learning_rate : 0.001)
#  loss : 0.00976622011512518
# [7,8,9,10]의 결과 :  [[10.65903]]

# 결과 (learning_rate : 0.009)
#  loss : 0.00019690097542479634
# [7,8,9,10]의 결과 :  [[10.892224]]


















