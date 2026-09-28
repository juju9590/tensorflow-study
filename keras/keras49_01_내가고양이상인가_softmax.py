# 실습 acc : 0.77


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import time
import datetime
from sklearn.metrics import accuracy_score

# 1. 데이터
start_data = time.time()

# 데이터 불러오기
data_path ="./_save/image/cat_dog/"

x_train = np.load(data_path+'cat_dog_x_train.npy')
y_train = np.load(data_path+'cat_dog_y_train.npy')
x_test = np.load(data_path+'cat_dog_x_test.npy')
y_test = np.load(data_path+'cat_dog_y_test.npy')

# 내사진 불러오기
img_path = './_save/my_photo/'

tori_111 = np.load(img_path + 'tori_111.npy')
yj_1 = np.load(img_path + 'yj_1.npy')
yj_2 = np.load(img_path + 'yj_2.npy')
yj_3 = np.load(img_path + 'yj_3.npy')

# 불러온 내 사진 스케일링 (rescale=1./255 은 스케일링된_사진 = 원본_사진 × (1./255) 뜻)

tori_111 = tori_111*(1./255)
yj_1 = yj_1*(1./255)
yj_2 = yj_2*(1./255)
yj_3 = yj_3*(1./255)

end_data = time.time()


# 2. 모델구성

start_time = time.time()

# 전체 모델 저장
model_path ='./_save/image/cat_dog/'
filename = 'cat_dog_model_0928_2th.keras'

model = load_model(model_path + filename)

end_time = time.time()

# 4. 평가, 예측
results = model.evaluate(x_test, y_test,)
print('loss : ', round(results[0],4))
print('acc : ', round(results[1],4))

y_pred = np.argmax(model.predict(x_test), axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", round(acc_score,4) ) 

print("========== 예측하기 ===========")

tori_111_raw = model.predict(tori_111)
yj_1_raw = model.predict(yj_1)
yj_2_raw = model.predict(yj_2)
yj_3_raw = model.predict(yj_3)

tori_111_pred = np.argmax(model.predict(tori_111), axis=1) 
yj_1_pred = np.argmax(model.predict(yj_1), axis=1) 
yj_2_pred = np.argmax(model.predict(yj_2), axis=1) 
yj_3_pred = np.argmax(model.predict(yj_3), axis=1) 


print("토리 softmax :", tori_111_raw, "토리:", tori_111_pred)
print("영주1 softmax :", yj_1_raw, "영주 1:", yj_1_pred)
print("영주2 softmax :", yj_2_raw, "영주 2:", yj_2_pred)
print("영주3 softmax :", yj_3_raw, "영주 3:", yj_3_pred)


print('데이터 불러오기 : ', round(end_time-start_time,3), "초")
print('모델/훈련 불러오기 : ', round(end_data-start_data,3), "초")

# #{'cats': 0, 'dogs': 1}

# 'cat_dog_model_0928.keras'
# loss :  0.4566
# acc :  0.7958
# acc_score :  0.7958
# 토리 softmax : [[0.04549257 0.9545074 ]] 토리: [1]
# 영주1 softmax : [[0.70796514 0.29203492]] 영주 1: [0]
# 영주2 softmax : [[0.1265657 0.8734343]] 영주 2: [1]
# 영주3 softmax : [[0.03582991 0.96417016]] 영주 3: [1]
# 데이터 불러오기 :  1.398 초
# 모델/훈련 불러오기 :  1.07 초

# 'cat_dog_model_0928_2th.keras'
# loss :  0.4365
# acc :  0.8403
# acc_score :  0.8403
# 토리 softmax : [[0.00129572 0.9987043 ]] 토리: [1]
# 영주1 softmax : [[0.8038695  0.19613048]] 영주 1: [0]
# 영주2 softmax : [[0.00222212 0.9977779 ]] 영주 2: [1]
# 영주3 softmax : [[0.04480848 0.9551915 ]] 영주 3: [1]
# 데이터 불러오기 :  1.362 초
# 모델/훈련 불러오기 :  1.059 초





