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


model_list =[VGG19(include_top=False, input_shape=(32,32,3)), 
             ResNet50(include_top=False, input_shape=(32,32,3)), 
             ResNet101(include_top=False, input_shape=(32,32,3)), 
             DenseNet201(include_top=False, input_shape=(32,32,3)), 
            # InceptionV3(), 
            # InceptionResNetV2(), # 개 고양이, 남자 여자 적용
             MobileNetV2(include_top=False, input_shape=(32,32,3)),
             EfficientNetB0(include_top=False, input_shape=(32,32,3)),
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


from tensorflow.keras.datasets import cifar10, cifar100

for model_i in model_list:
    model_i.trainable = False  # 모델 동결

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #01. 데이터
    (x_train, y_train), (x_test, y_test)= cifar10.load_data()

    # 스케일링 
    x_train = x_train/255.0
    x_test = x_test/255.0

    #3. 컴파일, 훈련
    from tensorflow.keras.optimizers import Adam

    # learning_rate = 0.01
    learning_rate = 0.001 # 디폴트 
    # learning_rate = 0.0001

    model.compile(
        loss="sparse_categorical_crossentropy", 
        optimizer=Adam(learning_rate=learning_rate),
        metrics = ['acc'])

    es = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=50,
        verbose=1,
        restore_best_weights=True,
    )

    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        mode='auto',
        patience=20,
        verbose=1,
        factor=0.5,
    )
    start_time=time.time()
    model.fit(x_train,y_train,
        epochs=100, 
        batch_size=128, 
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

# 결과
# 모델명 : vgg19 loss : 1.2136341333389282 acc : 0.5792999863624573




   





