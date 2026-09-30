# 19-4 카피

import numpy as np # 수치 계산에 특화
import pandas as pd # sklearn 만큼 강력함
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
path = "./_data/ddarung/" # 상대경로
# path = "c:/study/_data/ddarung/" # 절대경로
# path = "c:\study\_data\ddarung/" # 슬래시 역슬래시 / 2개 상관없음, 섞어쓰기 되지만 가급적 비권장
# path = "c://study//_data//ddarung/"
# path = "c:\\study\\_data\\ddarung/"

# train_csv는 x,y(count) 가 합쳐진 상태로, 분리해줘야함
train_csv = pd.read_csv(path + "train.csv", index_col=0) # csv 가져오기, 인덱스 컬럼 데이터에 포함 x 
print(train_csv) # id열 포함 [1459 rows x 11 columns] -> id열 안 포함 [1459 rows x 10 columns] index_col=0 때문

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv) # [715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission) # [715 rows x 1 columns]

# 데이터 받으면 shape 찍기
print(train_csv.shape) # (1459, 10) // 훈련에서 분리해야 하는 데이터 train_test_split
print(test_csv.shape) # (715, 9) // 제출용 파일, y 없음
print(submission.shape) # (715, 1)

print(train_csv.columns) # 컬럼이 중요함
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='str')

print(train_csv.info())
print(test_csv.info())

# exit()
############################# 결측치 처리 1. 삭제 ################################
# 데이터가 많을 때는 삭제 고려 가능
# 모델 성능이 좋으려면 데이터를 최대한 살려야함
train_csv = train_csv.dropna() # 결측치 있는 row 자체를 없애버림
print(train_csv) # [1328 rows x 10 columns]

# train_csv를 x와 y로 분리
x = train_csv.drop(['count'], axis=1) # 열(컬럼) 삭제 | 행0, 열1
print(x) # [1328 rows x 9 columns]

y = train_csv['count'] # pandas에서 컬럼만 빼는거
print(y)
print(y.shape) # (1328,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=333)




from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()



scaler.fit(x_train) 

x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test))



# 2. 모델 구성
model = Sequential()
model.add(Dense(8, input_dim=9))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")

import time
start_time = time.time()
hist = model.fit(x_train, y_train, 
                epochs=500, batch_size=160,
                validation_split=0.2,
                )
end_time = time.time()

print("걸린시간 :", round(end_time-start_time,2), "초")

print("==============================================")

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)
print("r2 : ", r2_score(y_test, y_pred))
print("rmse : ", np.sqrt(mean_squared_error(y_test, y_pred)))

# print("================== history ===============================")
# print(hist)
# print("================== hist.history ==========================")
# print(hist.history)
# print("================== loss ==========================")
# print(hist.history['loss'])
# print("================== val_loss ==========================")
# print(hist.history['val_loss'])

# print("================== 시각화 ==========================")
# import matplotlib.pyplot as plt

# # plt에서 한글 못 읽기 때문에 맑은고딕 폰트 설정 필수
# # plt.rcParams['font.family']='Malgun Gothic'
# plt.rc('font', family = 'Hancom Gothic')

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'][30:], c='red', label='loss')
# plt.plot(hist.history['val_loss'][30:], c='blue', label='val_loss')
# # x를 명시하지 않으면 y값을 시간순으로 그려줌
# plt.legend(loc="upper right") # 우측 상단에 라벨표시(범례)
# plt.title('따릉이 Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid() #모눈종이처럼 표시(격자)
# plt.show()

############################ 평가 ##################
# Epoch 500/500
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 3090.9402 - val_loss: 3062.1284
# ==============================================
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 651us/step - loss: 2702.7158
# loss :  2702.7158203125
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# r2 :  0.6247159344553991
# rmse :  51.98765227405579

######################### MinMaxScaler ############### ==> 아주아주 소폭 향상
# Epoch 500/500
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 2864.2085 - val_loss: 3094.8616
# ==============================================
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 2666.0181 
# loss :  2666.01806640625
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# r2 :  0.6298115715996777
# rmse :  51.633499731080285

######################### StandardScaler ############### ==>  향상
# Epoch 500/500
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 2581.3792 - val_loss: 2822.3831
# 걸린시간 : 26.94 초
# ==============================================
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 2389.9885
# loss :  2389.988525390625
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# r2 :  0.6681395133649817
# rmse :  48.887509786399406


######################### MaxAbscaler ############### ==>  loss 하향, r2, rmse 향상
# Epoch 500/500
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 2918.9011 - val_loss: 3131.4092
# 걸린시간 : 27.03 초
# ==============================================
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 2709.3752 
# loss :  2709.375244140625
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# r2 :  0.62379127965514
# rmse :  52.05165853357736

######################### RobustScaler ############### ==>  loss 성능개선, R2, Rmse 개선
# Epoch 500/500
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 2630.2822 - val_loss: 2824.8733
# 걸린시간 : 27.45 초
# ==============================================
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 785us/step - loss: 2383.5959
# loss :  2383.595947265625
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# r2 :  0.6690272155320054
# rmse :  48.82208077283538