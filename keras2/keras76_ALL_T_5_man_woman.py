import warnings
warnings.filterwarnings("ignore")  #경고메시지 무시 

from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101,ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet169, DenseNet121
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetMobile, NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications import Xception


model_list =[VGG19(include_top=False, input_shape=(100,100,3)), 
             ResNet50(include_top=False, input_shape=(100,100,3)), 
             ResNet101(include_top=False, input_shape=(100,100,3)), 
             DenseNet201(include_top=False, input_shape=(100,100,3)), 
             InceptionV3(include_top=False, input_shape=(100,100,3)), 
             InceptionResNetV2(include_top=False, input_shape=(100,100,3)),
             MobileNetV2(include_top=False, input_shape=(100,100,3)),
             EfficientNetB0(include_top=False, input_shape=(100,100,3)),
             ]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score
import random
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)


for model_i in model_list:
    model_i.trainable = False  # 모델 동결

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    # model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    #01. 데이터
    np_path = './_save/keras44/'

    # 불러오기
    x_train = np.load(np_path + 'keras44_03_men_women_x_train.npy')
    y_train = np.load(np_path + 'keras44_03_men_women_y_train.npy')
    x_test = np.load(np_path + 'keras44_03_men_women_x_test.npy')
    y_test = np.load(np_path + 'keras44_03_men_women_y_test.npy')

    print(x_train.shape, y_train.shape) #
    print(x_test.shape, y_test.shape) #

    #3. 컴파일, 훈련
    from tensorflow.keras.optimizers import Adam

    # learning_rate = 0.01
    learning_rate = 0.001 # 디폴트 
    # learning_rate = 0.0001

    model.compile(
        loss='binary_crossentropy', 
        optimizer=Adam(learning_rate=learning_rate),
        metrics = ['acc'])

    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=50,
        verbose=1,
        restore_best_weights=True,
    )

    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        patience=20,
        verbose=1,
        factor=0.5,
    )
    start_time=time.time()
    model.fit(x_train,y_train,
        epochs=100, 
        batch_size=32, 
        verbose=0,
        validation_split=0.2,
        callbacks = [es, rlr],
        )
    end_time=time.time()
    print ('모델/훈련 걸린시간 : ', round(end_time-start_time,3),'초' )

    #04. 평가, 예측
    results = model.evaluate(x_test, y_test, verbose=0,)
    print("===========================================")
    print("모델명 :", model_i.name, "loss :", results[0], "acc :", results[1])


# 결과 2차 (GAP) 모델3
# loss :  0.278
# acc :  0.928
# acc_score :  0.928
# 걸린시간 :  662.285 초


    






   





