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


model_list =[VGG19(include_top=False, input_shape=(150,150,3)), 
             ResNet50(include_top=False, input_shape=(150,150,3)), 
             ResNet101(include_top=False, input_shape=(150,150,3)), 
             DenseNet201(include_top=False, input_shape=(150,150,3)), 
            # InceptionV3(), 
            # InceptionResNetV2(), # 개 고양이, 남자 여자 적용
             MobileNetV2(include_top=False, input_shape=(150,150,3)),
             EfficientNetB0(include_top=False, input_shape=(150,150,3)),
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
    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    #01. 데이터
    np_path = './_save/keras46_02/' 

    # 불러오기
    x_train = np.load(np_path + 'keras46_02_rps_x_train.npy')
    y_train = np.load(np_path + 'keras46_02_rps_y_train.npy')
    x_test = np.load(np_path + 'keras46_02_rps_x_test.npy')
    y_test = np.load(np_path + 'keras46_02_rps_y_test.npy')

    print(x_train.shape, y_train.shape) #(1433, 150, 150, 3) (1433, 3)
    print(x_test.shape, y_test.shape) #(615, 150, 150, 3) (615, 3)

    #3. 컴파일, 훈련
    from tensorflow.keras.optimizers import Adam

    # learning_rate = 0.01
    learning_rate = 0.001 # 디폴트 
    # learning_rate = 0.0001

    model.compile(
        loss="categorical_crossentropy", 
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


#####  VGG16 동결모드 epochs= 50, batch_size=32,
# loss :  0.0
# acc :  1.0
# acc_score :  1.0
# 걸린시간 :  121.615 초

# 모델/훈련 걸린시간 :  277.452 초
# ===========================================
# 모델명 : vgg19 loss : 1.0654619472916238e-05 acc : 1.0

# 모델/훈련 걸린시간 :  142.417 초
# ===========================================
# 모델명 : resnet50 loss : 0.00014290993567556143 acc : 1.0

# 모델/훈련 걸린시간 :  232.248 초
# ===========================================
# 모델명 : resnet101 loss : 0.0007682904251851141 acc : 1.0

# 모델/훈련 걸린시간 :  208.925 초
# ===========================================
# 모델명 : densenet201 loss : 7.622185421496397e-06 acc : 1.0

# 모델/훈련 걸린시간 :  60.488 초
# ===========================================
# 모델명 : mobilenetv2_1.00_224 loss : 6.688711891911225e-06 acc : 1.0

# 모델/훈련 걸린시간 :  85.578 초
# ===========================================
# 모델명 : efficientnetb0 loss : 1.033836841583252 acc : 0.6195122003555298
    






   





