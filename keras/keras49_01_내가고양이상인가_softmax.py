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
filename = 'cat_dog_model.keras'

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

tori_111_pred = np.argmax(model.predict(tori_111), axis=1) 
yj_1_pred = np.argmax(model.predict(yj_1), axis=1) 
yj_2_pred = np.argmax(model.predict(yj_2), axis=1) 
yj_3_pred = np.argmax(model.predict(yj_3), axis=1) 


print("토리:", tori_111_pred)
print("영주 1:", yj_1_pred)
print("영주 2:", yj_2_pred)
print("영주 3:", yj_3_pred)


print('데이터 불러오기 : ', round(end_time-start_time,3), "초")
print('모델/훈련 불러오기 : ', round(end_data-start_data,3), "초")










