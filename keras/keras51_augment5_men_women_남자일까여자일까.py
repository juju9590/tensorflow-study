# 데이터 증강 실습

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import time
import datetime
from sklearn.metrics import accuracy_score

# 1. 데이터
start_data = time.time()

# 최종 데이터 불러오기(경로 및 파일네임)
data_aug_path = './_save/image/men_women/'
data_aug_filename = 'men_woman_aug_1005_'
data_path = './_save/image/men_women/'
data_filename = 'men_women_1005_'

x_train = np.load(data_aug_path + data_aug_filename + 'x_train.npy' )
y_train = np.load(data_aug_path + data_aug_filename + 'y_train.npy' )
x_test = np.load(data_path + data_filename + 'x_test.npy')
y_test = np.load(data_path + data_filename + 'y_test.npy')

end_data = time.time()
print('data 걸린시간 :', round(end_data-start_data, 3),"초")

#2. 모델구성
start_model = time.time()

# 모델 저장 경로 및 이름
model_aug_path = './_save/image/men_women/'
model_aug_filename = 'men_womne_aug_1005_model.keras'

# 전체 모델 불러오기
model = load_model(model_aug_path + model_aug_filename)

end_model = time.time()
print('모델/훈련 걸린시간 :', round(end_model-start_model, 3),"초")

#4. 평가, 예측
start_eva = time.time()

results = model.evaluate(x_test, y_test)
print("loss : ", round(results[0],3))
print("acc : ", round(results[1],3))

end_eva = time.time()
print('평가 걸린시간 :', round(end_eva-start_eva, 3),"초")

start_pred = time.time()

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_pred, y_test)
print("acc_score : ", round(acc_score,3))

end_pred = time.time()
print('예측/정확도 걸린시간 :', round(end_eva-start_eva, 3),"초")

######################################################
# 예측 사진 준비
photo_path = './_data/my_photo/'

# 이미지 1장 씩 불러오기
yj_2 = load_img( photo_path + 'yj_2.jpeg',target_size=(150,150),)
tori_111 = load_img(photo_path + 'tori_111.jpeg', target_size=(150,150),)

# 이미지 1장 씩 수치화 하기 + 스케일링 (0~1사이로 만들어주기)
yj_2 = img_to_array(yj_2)/255.0
tori_111 = img_to_array(tori_111)/255.0

print(np.min(yj_2),np.max(yj_2)) #0.0 1.0
print(np.min(tori_111),np.max(tori_111)) #0.0 1.0

print(yj_2.shape, tori_111.shape) #(150, 150, 3) (150, 150, 3)
print('x_test.shape :', x_test.shape) #(8151, 150, 150, 3)

# 차원증가 (3차원  -> 4차원)
yj_2 = np.expand_dims(yj_2, axis=0)
tori_111 = np.expand_dims(tori_111, axis=0)

print(yj_2.shape, tori_111.shape) #(1, 150, 150, 3) (1, 150, 150, 3)

print("====== 내 사진 예측 ======")
y_pred_me = model.predict(yj_2)
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



# 결론 (저장)
# data 걸린시간 : 70.325 초
# 모델/훈련 걸린시간 : 236.636 초
# loss :  0.397
# acc :  0.823
# acc_score :  0.823

# 결론 (불러오기)
# data 걸린시간 : 9.307 초
# 모델/훈련 걸린시간 : 0.511 초
# loss :  0.397
# acc :  0.823
# 평가 걸린시간 : 33.765 초
# acc_score :  0.823
# 예측/정확도 걸린시간 : 33.765 초

# ====== 내 사진 예측 ======
# 성별 : [1] 내 사진 : [[0.5464535]]
# 내 사진 예측 : [[1.]] ===> 여자
# 성별 : [0] 토리 사진 : [[0.5304389]]
# 토리 사진 예측 : [[1.]] ===>토리는 고양이, 여자로 나옴


# 결과 2차
# data 걸린시간 : 24.987 초
# 모델/훈련 걸린시간 : 3.121 초
# loss :  0.914
# acc :  0.868
# 평가 걸린시간 : 13.962 초
# acc_score :  0.868
# 예측/정확도 걸린시간 : 13.962 초
# ====== 내 사진 예측 ======
# 성별 : [1] 내 사진 : [[1.2108638e-11]]
# 내 사진 예측 : [[0.]]
# 성별 : [0] 토리 사진 : [[4.8231443e-05]]
# 토리 사진 예측 : [[0.]]