# 11-2 copy
# 파라미터는 성능향상을 위해 조정하는것

from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    test_size=0.3,
    random_state=389,
    shuffle=True
)

#2. 모델 (함수형을 함수로 만들기)
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(10,), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)

    model = Model(inputs=inputs,outputs=outputs)

    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')
    return model
    # 이름 똑같이 해도 상관없음, 위 -> 아래로 실행되기 때문에 = Sequential() 구조에서는 가능
    # 계층마다 node 

def create_hyperparameter():
    batchs = [32,16,8,1,64]
    optimizers =['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu','elu','selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16]
    return {
        'batch_size': batchs,
        'optimizer' : optimizers,
        'drop' : dropouts,
        'activation' : activations,
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'node4' : node4,
        'node5' : node5,
    }

# {'batch_size': [32, 16, 8, 1, 64], 
# 'optimizer': ['adam', 'rmsprop', 'adadelta'], 
# 'drop': [0.2, 0.3, 0.4, 0.5], 
# 'activation': ['relu', 'elu', 'selu', 'linear'], 
# 'node1': [128, 64, 32, 16], 
# 'node2': [128, 64, 32, 16], 
# 'node3': [128, 64, 32, 16], 
# 'node4': [128, 64, 32, 16], 
# 'node5': [128, 64, 32, 16]}

hyperparameters = create_hyperparameter() # 딕셔너리가 들어감
print(hyperparameters)

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# 케라스 모델을 사이킷런 랜덤서치에서 모델을 인식 못함, 케라스 모델을 사이킷런 형태로 바꿔준다

keras_model = KerasRegressor(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameters, cv=3, 
                           random_state=333, 
                           n_iter=10, 
                           verbose=1,
                           refit=True,
                        #    n_jobs=-1,
                           )

# 03. 훈련
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=8,
    verbose=1,
    factor=0.8,
)


start = time.perf_counter()
model.fit(x_train, y_train, 
          epochs=1000, 
          verbose=1, 
        #   callbacks=[es, rlr],
          validation_split=0.1,)
end = time.perf_counter()
print("==========================================")
print("최적의 매개변수 :", model.best_estimator_) #estimator=평가자
print("==========================================")
print("최적의 파라미터 :", model.best_params_)
print("==========================================")

#04. 평가, 예측
print("==========================================")
print('model.socore :',model.score(x_test, y_test))
print('best_score :', model.best_score_) #train 데이터에 대한 스코어 (과적합이 발생함)

print('걸린시간 :', round(end-start,2),'초')

# RandomizedSearchCV///KerasRegressor///1차

# 최적의 매개변수 : <keras.wrappers.scikit_learn.KerasRegressor object at 0x00000279432D1D80>
# 최적의 파라미터 : {'optimizer': 'adam', 'node5': 32, 'node4': 32, 'node3': 16, 'node2': 32, 'node1': 32, 'drop': 0.4, 'batch_size': 8, 'activation': 'selu'}
# 17/17 [==============================] - 0s 1ms/step - loss: 2908.4353 - mae: 42.5479
# model.socore : -2908.435302734375
# best_score : -3090.2432454427085
# 걸린시간 : 298.77 초

# Epoch 68: early stopping (epochs=100 )
# ==========================================
# 최적의 매개변수 : <keras.wrappers.scikit_learn.KerasRegressor object at 0x0000015A3280C610>
# ==========================================
# 최적의 파라미터 : {'optimizer': 'rmsprop', 'node5': 32, 'node4': 64, 'node3': 32, 'node2': 16, 'node1': 16, 'drop': 0.3, 'batch_size': 8, 'activation': 'selu'}
# ==========================================
# ==========================================
# 17/17 [==============================] - 0s 8ms/step - loss: 2831.2725 - mae: 41.9950
# model.socore : -2831.2724609375
# best_score : -3173.442626953125
# 걸린시간 : 246.59 초


