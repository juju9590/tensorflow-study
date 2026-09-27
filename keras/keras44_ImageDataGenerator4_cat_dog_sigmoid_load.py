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

start_data =time.time()

# 데이터 저장하기 

data_path="./_save/image/cat_dog/"

x_train = np.load(data_path + 'cat_dog_sigmoid_x_train.npy')
y_train = np.load(data_path + 'cat_dog_sigmoid_y_train.npy')
x_test = np.load(data_path + 'cat_dog_sigmoid_x_test.npy')
y_test = np.load(data_path + 'cat_dog_sigmoid_y_test.npy')

end_data =time.time()


# 2. 모델구성
start_time = time.time()

##### 모델 저장
model_path ='./_save/image/cat_dog/'
filename = 'cat_dog_sigmoid_model.keras' # 확장자 : .keras

model = load_model(model_path + filename)

end_time = time.time()


# 4. 평가, 예측
results = model.evaluate(x_test, y_test,)
print('loss : ', round(results[0],3))
print('acc : ', round(results[1],3))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 반올림 처리

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score ) 

print('데이터 걸린시간 : ', round(end_time-start_time,3), "초")
print('훈련 걸린시간 : ', round(end_data-start_data,3), "초")


# 결과 6
# loss :  3.146
# acc :  0.784
# acc_score :  0.7839841819080573
# 걸린시간 :  628.565 초


### sigmoid (epochs= 1)
# Found 8005 images belonging to 2 classes.
# Found 2023 images belonging to 2 classes.
# (8005, 150, 150, 3) (8005,)
# (2023, 150, 150, 3) (2023,)
# loss :  0.693
# acc :  0.503
# acc_score :  0.5027187345526446
# 데이터 걸린시간 :  80.541 초
# 훈련 걸린시간 :  49.453 초

### sigmoid (epochs= 1) 데이터저장, 모델저장 후 불러오기
# loss :  0.693
# acc :  0.503
# acc_score :  0.5027187345526446
# 데이터 걸린시간 :  0.277 초
# 훈련 걸린시간 :  1.963 초











