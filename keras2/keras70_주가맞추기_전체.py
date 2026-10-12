
# 삼성전자 1013 시가 맞추기
# 앙상블 쓸것
# LSTM, Conv1D 쓸것

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU, Input
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam


#1. 데이터

### 원본 데이터 가져오기
path="./_data/samsung/"
x1 = pd.read_csv(path+'아모레261012.csv',  thousands=',', index_col=0, encoding='cp949')
x2 = pd.read_csv(path+'하이닉스261012.csv', thousands=',', index_col=0, encoding='cp949')
x3 = pd.read_csv(path+'삼성전자261012.csv', thousands=',', index_col=0, encoding ='cp949')

# print(x1) #[2060 rows x 16 columns]
# print(type(x1)) #<class 'pandas.core.frame.DataFrame'>

# print(x1.columns) # 컬럼이 중요함
# Index(['시가', '고가', '저가', '종가', '대비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
#        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
#       dtype='object')

# 행 맞추기
x1_data = x1[:-940].drop(['시가','대비', 'Unnamed: 6',], axis=1)
x2_data = x2[:-180].drop(['시가','대비', 'Unnamed: 6', ], axis=1)
x3_data = x3.drop(['시가','대비', 'Unnamed: 6',], axis=1) # x,y 분리

x1_predict = x1_data[:6]
x2_predict = x2_data[:6]
x3_predict = x3_data[:6]

# print(x1_data.shape) #(1120, 14)
# print(x2_data.shape) #(1120, 14)
# print(x3_data.shape) #(1120, 13)

y_data = x3["시가"]
# print(y_data.shape) #(1120,)
y_true = y_data[:1]


# train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1_data, x2_data,x3_data,
    y_data,
    train_size=0.8,
    random_state=333,
    shuffle=False,              # 시간순유지, 섞이면 안됨
)
# print(x1_train.shape, x1_test.shape) #(896, 14) (224, 14)
# print(x2_train.shape, x2_test.shape) #(896, 14) (224, 14)
# print(x3_train.shape, x3_test.shape) #(896, 13) (224, 13)
# print(y_train.shape, y_test.shape) #(896,) (224,)

# 학습용 시계열 데이터 만들기
size = 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

x1_train = split_x(x1_train, size)
x1_test = split_x(x1_test, size)
x2_train = split_x(x2_train, size)
x2_test = split_x(x2_test, size)
x3_train = split_x(x3_train, size)
x3_test = split_x(x3_test, size)

y_train = split_x(y_train, size)
y_test = split_x(y_test, size)

# print(x1_train.shape) # (891, 6, 14)
# print(x1_test.shape) # (219, 6, 14)
# print(x2_train.shape) # (891, 6, 14)
# print(x2_test.shape) # (219, 6, 14)
# print(x3_train.shape) # (891, 6, 13)
# print(x3_test.shape) # (219, 6, 13)
# print(y_train.shape) # (891, 6)
# print(y_test.shape) # (219, 6)

end_split=time.time()

# x_train 스케일링 (3차원 ==> 2차원 변경 후 스케일링 ==> 3차원 원복)
scaler = MinMaxScaler()

# 2차원으로 변환
x1_train = x1_train.reshape(-1, 13)
x1_test = x1_test.reshape(-1, 13)
x2_train = x2_train.reshape(-1, 13)
x2_test = x2_test.reshape(-1, 13)
x3_train = x3_train.reshape(-1, 13)
x3_test = x3_test.reshape(-1, 13)

# print(x1_train.shape) # (5346, 14)
# print(x1_test.shape) # (1314, 14)
# print(x2_train.shape) # (5346, 14)
# print(x2_test.shape) # (1314, 14)
# print(x3_train.shape) # (5346, 13)
# print(x3_test.shape) # (1314, 13)

x1_train = scaler.fit_transform(x1_train)  
x1_test = scaler.transform(x1_test)  
x2_train = scaler.fit_transform(x2_train)  
x2_test = scaler.transform(x2_test)  
x3_train = scaler.fit_transform(x3_train)  
x3_test = scaler.transform(x3_test)  

# print(np.min(x1_train), np.max(x1_train)) #0.0 1.0
# print(np.min(x2_train), np.max(x2_train)) #0.0 1.0
# print(np.min(x3_train), np.max(x3_train)) #0.0 1.0

x1_predict = scaler.transform(x1_predict)
x2_predict = scaler.transform(x2_predict)
x3_predict = scaler.transform(x3_predict)

# RNN계열 모델 적용 위해  3차원으로 변경
x1_train = x1_train.reshape(-1, 6, 13)
x1_test = x1_test.reshape(-1, 6, 13)
x2_train = x2_train.reshape(-1, 6, 13)
x2_test = x2_test.reshape(-1, 6, 13)
x3_train = x3_train.reshape(-1,6, 13)
x3_test = x3_test.reshape(-1, 6, 13)

# print(x1_train.shape) # (891, 6, 14)
# print(x1_test.shape) # (219, 6, 14)
# print(x2_train.shape) # (891, 6, 14)
# print(x2_test.shape) # (219, 6, 14)
# print(x3_train.shape) # (891, 6, 13)
# print(x3_test.shape) # (219, 6, 13)

x1_predict = x1_predict.reshape(-1,6,13)
x2_predict = x2_predict.reshape(-1,6,13)
x3_predict = x3_predict.reshape(-1,6,13)

### y_train, y_test 도  shape 맞추기
# y_train = y_train.reshape(-1,6,1)
# y_test = y_test.reshape(-1,6,1)
# print(y_train.shape) # (891, 6, 1)
# print(y_test.shape) #(219, 6, 1)

#02. 모델구성

#02-1 모델구성
input1 = Input(shape=(6,13)) 
dense1 = LSTM(64, activation='relu', return_sequences=True, name='han_1')(input1) 
dense2 = LSTM(64, activation='relu', return_sequences=True, name='han_2')(dense1) 
dense3 = Conv1D(32, 3, activation='relu', name='han_3')(dense2) 
dense4 = Flatten() (dense3) 
output1 = Dense(16, activation='relu', name='han_4')(dense4) 


#02-2 모델구성
input21 = Input(shape=(6,13)) 
dense21 = LSTM(64, activation='relu', return_sequences=True, name='han_21')(input21) 
dense22 = LSTM(64, activation='relu', return_sequences=True, name='han_22')(dense21) 
dense23 = Conv1D(32, 3, activation='relu', name='han_23')(dense22) 
dense24 = Conv1D(32, 3, activation='relu', name='han_24')(dense23) 
dense25 = Flatten() (dense24) 
output21 = Dense(16, activation='relu', name='han_25')(dense25) 

#02-3 모델구성
input31 = Input(shape=(6,13)) 
dense31 = Conv1D(32, 2, activation='relu', name='han_31')(input31) 
dense32 = LSTM(32, activation='relu', return_sequences=True, name='han_32')(dense31) 
dense33 = LSTM(32, activation='relu', return_sequences=True, name='han_33')(dense32)
dense34 = Flatten() (dense33) 
output31 = Dense(16, activation='relu', name='han_34')(dense34) 


#02-4 모델 합치기
merge_1 = concatenate([output1, output21, output31 ], name="mg_1")
# merge_2 = Dense(10, name='mg_2')(merge_1)
# merge_3 = Dense(5, name='mg_3')(merge_2)
last_output = Dense(1, name='last')(merge_1)
model = Model(inputs=[input1, input21, input31], outputs=last_output)
model.summary()

#3. 컴파일, 훈련

learning_rate = 0.001 # 디폴트 

model.compile(
    loss="mse", 
    optimizer=Adam(learning_rate=learning_rate),
)

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=40,
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

model.fit([x1_train, x2_train, x3_train], [y_train], 
    epochs=1, 
    batch_size=32,
    # verbose=1,
    # callbacks = [es, rlr, ],
    validation_split = 0.2,
    )

# 전체 모델 저장
# model_path = "./_save/kaggle_jena/"
# filename = 'jena_model_Conv1D.keras'

# model.save(model_path + filename)

# 모델 불러오기
# model = load_model(model_path + filename)

#4. 평가, 예측
results = model.evaluate([x1_test, x2_test, x3_test], y_test)
print('loss :', results) 

y_pred = model.predict([x1_predict, x2_predict, x3_predict])
# print("x1 :", y_pred[0], "x2 : ", y_pred[1], "x3 : ", y_pred[2])
print(y_pred)


