
# 실습 :  acc 0.94~97

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_load, img_to_array

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import time
import datetime
from sklearn.metrics import accuracy_score


# 1. 데이터

np_path = './_save/keras44/'

# 불러오기
x_train = np.load(np_path + 'keras44_03_men_women_x_train.npy')
y_train = np.load(np_path + 'keras44_03_men_women_y_train.npy')
x_test = np.load(np_path + 'keras44_03_men_women_x_test.npy')
y_test = np.load(np_path + 'keras44_03_men_women_y_test.npy')


# 내 사진 불러오기
np_path = './_data/kaggle_cat_dog_npy/'

my_photo = np.load(np_path + 'keras48_my.npy')
tori_photo = np.load(np_path + 'keras48_tori.npy')

# 스케일링
my_photo = my_photo/255.
tori_photo = tori_photo/ 255

print(np.max(my_photo), np.min(my_photo))  # 1.0 0.011764706
print(np.max(tori_photo), np.min(tori_photo)) #1.0 0.0

# exit()

# 2. 모델구성 + #3. 컴파일, 훈련

path ='./_save/keras49/'
model = load_model(path + 'keras49_save_model_2.keras')


# 4. 성능, 예측
results = model.evaluate(x_test, y_test,)
print('loss : ', round(results[0],4))
print('acc : ', round(results[1],4))

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=1)
y_test_arg = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test_arg, y_pred)

print("acc_score :", acc_score)


print("=========== 내 사진 예측===============")

y_pred_me = model.predict(my_photo)

print(y_pred_me)

pred_class = np.argmax(y_pred_me, axis=1)

print("예측 클래스 :", pred_class)



print("=========== 토리 예측===============")

y_pred_tori = model.predict(tori_photo)
y_pred_tori = np.round(y_pred_tori,4) # 반올림 처리

print(y_pred_tori, y_pred_tori.shape ) #

# 결과
# loss :  0.242
# acc :  0.9222
# acc_score : 1.0
# =========== 내 사진 예측===============
# [[0.00548798]]
# 예측 클래스 : [0]
# =========== 토리 예측===============
# [[0.2299]] (1, 1)