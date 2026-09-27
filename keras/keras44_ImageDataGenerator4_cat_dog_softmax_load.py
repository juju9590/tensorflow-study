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

print("데이터 변환시간 : ", round(end_data-start_data,2),"초")
print('훈련 걸린시간 : ', round(end_time-start_time,2), "초")


# 결과 6
# loss :  3.146
# acc :  0.784
# acc_score :  0.7839841819080573
# 걸린시간 :  628.565 초

# softmax (epochs=1)
# Found 8005 images belonging to 2 classes.
# Found 2023 images belonging to 2 classes.
# (8005, 150, 150, 3) (8005, 2)
# (2023, 150, 150, 3) (2023, 2)
# 데이터 걸린시간 :  34.9 초
### ecope 1번
# loss :  0.6931
# acc :  0.4998
# acc_score :  0.49975284231339595
# 훈련 걸린시간 :  183.28 초

### softmax_load (epochs=1)
# loss :  0.6931
# acc :  0.4998
# acc_score :  0.49975284231339595
# 데이터 변환시간 :  1.89 초
# 훈련 걸린시간 :  0.25 초









