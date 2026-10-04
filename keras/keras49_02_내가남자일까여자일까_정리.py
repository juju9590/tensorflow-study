
# 실습 :  acc 0.94~97

import numpy as np
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D,Dense,Flatten,GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#1. 데이터

start_data = time.time()

# 데이터 저장 경로
data_path = './_save/image/men_women/'
data_filename = 'men_women_1004_'

# 데이터 불러오기
x_train = np.load(data_path + data_filename + 'x_train.npy')
y_train = np.load(data_path + data_filename + 'y_train.npy')
x_test = np.load(data_path + data_filename + 'x_test.npy')
y_test = np.load(data_path + data_filename + 'y_test.npy')

end_data = time.time()
print('data 걸린시간 :', round(end_data-start_data, 3),"초") 

#2. 모델 구성
start_model=time.time()

# 모델 저장 경로 및 이름
model_path ='./_save/image/men_women/'
filename = 'men_women_1004_'

# 모델 불러오기
model = load_model(model_path + filename + ".keras")

end_model=time.time()
print("모델/훈련 걸린시간 :", round(end_model-start_model,2),"초")

#4. 평가, 예측
print("====== 기본 ======")

results = model.evaluate(x_test, y_test)
print('loss : ', round(results[0],3))
print('acc : ', round(results[1],3))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) 

acc = accuracy_score(y_pred, y_test)
print("acc_score :", round(acc,3) )

#########################################################
# 예측 사진 준비
photo_path = './_data/my_photo/'

# 이미지 1장 씩 불러오기
yj_1 = load_img( photo_path + 'yj_2.jpeg',target_size=(150,150),)
tori_111 = load_img(photo_path + 'tori_111.jpeg', target_size=(150,150),)

# 이미지 1장 씩 수치화 하기 + 스케일링 (0~1사이로 만들어주기)
yj_1 = img_to_array(yj_1)/255.0
tori_111 = img_to_array(tori_111)/255.0

print(np.min(yj_1),np.max(yj_1)) #0.0 1.0
print(np.min(tori_111),np.max(tori_111)) #0.0 1.0

print(yj_1.shape, tori_111.shape) #(150, 150, 3) (150, 150, 3)
print('x_test.shape :', x_test.shape) #(8151, 150, 150, 3)

# 차원증가 (3차원  -> 4차원)
yj_1 = np.expand_dims(yj_1, axis=0)
tori_111 = np.expand_dims(tori_111, axis=0)

print(yj_1.shape, tori_111.shape) #(1, 150, 150, 3) (1, 150, 150, 3)

print("====== 내 사진 예측 ======")
y_pred_me = model.predict(yj_1)
y_true_me = np.array([1]) #정답 : 여자=1
print('성별 :', y_true_me, "내 사진 :", y_pred_me )

y_pred_me = np.round(y_pred_me)
acc = accuracy_score(y_pred_me, y_true_me)
print("내 사진 예측 :", y_pred_me)

# print("===== 토리 사진 예측 =====")
y_pred_tori = model.predict(tori_111)
y_true_tori = np.array([0]) #정답 : 남자=0
print('성별 :', y_true_tori, "토리 사진 :", y_pred_tori )

y_pred_tori = np.round(y_pred_tori) #

acc = accuracy_score(y_pred_tori, y_true_tori)
print("토리 사진 예측 :", y_pred_tori)


# 결과
# data 걸린시간 : 8.155 초
# 모델/훈련 걸린시간 : 0.41 초
# loss :  0.424
# acc :  0.797
# acc_score : 0.797

# ====== 내 사진 예측 ======yj_1.png
# 성별 : [1] 내 사진 : [[0.17475824]]
# 내 사진 예측 : [[0.]] ===> False
# 성별 : [0] 토리 사진 : [[0.5795522]]
# 토리 사진 예측 : [[1.]]

# ====== 내 사진 예측 ======yj_2.jpeg
# 성별 : [1] 내 사진 : [[0.760272]]
# 내 사진 예측 : [[1.]] ===> True











