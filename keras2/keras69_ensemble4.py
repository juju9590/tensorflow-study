# 예측값까지 완성

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, mean_squared_error


#01. 데이터
x1_datasets = np.array([range(100), range(301,401)]).T

y1 = np.array(range(3001,3101)) # 온도 
y2 = np.array(range(13001,13101)) # 비트코인가격


x1_pred = np.array([range(100,106), range(400,406)]).T

print(x1_pred.shape) #(6, 2)

# train_test_split
x1_train, x1_test, y1_train, y1_test , y2_train, y2_test= train_test_split(
    x1_datasets, y1, y2,
    train_size=0.7, 
    random_state=7,
    )

print(x1_train.shape, x1_test.shape) #(70, 2) (30, 2)
print(y1_train.shape, y1_test.shape) #(70,) (30,)
print(y2_train.shape, y2_test.shape) #(70,) (30,)

#02-1 모델구성

input1 = Input(shape=(2,)) 
dense1 = Dense(128, activation='relu', name='han_1')(input1) 
dense2 = Dense(128, activation='relu', name='han_2')(dense1) 
dense3 = Dense(64, activation='relu', name='han_3')(dense2) 
dense4 = Dense(64, name='han_4')(dense3) 
output1 = Dense(16, name='han_5')(dense4) 

#02-5 분기 1
# last_dense_1 = Dense(16, activation='relu', name='ld_1')(output1)
# last_dense_2 = Dense(16, activation='relu', name='ld_2')(last_dense_1)
last_output_1 = Dense(1, name='last1')(output1)

#02-6 분기 2
last_output_2 = Dense(1, name='last2')(output1)

model = Model(inputs=[input1], outputs=[last_output_1, last_output_2] )

model.summary()

#03. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# es = EarlyStopping(
#     monitor="val_loss",
#     mode='min',
#     patience=20,
#     restore_best_weights=True,
# )

model.fit([x1_train], [y1_train, y2_train],
    epochs=700,
    batch_size=32,
    validation_split=0.3,
    # callbacks = [es, ],
    )

#04. 평가, 예측
results = model.evaluate([x1_test], [y1_test, y2_test] )
print('loss :', results)

y_pred = model.predict([x1_pred])
print("온도 :", y_pred[0], "가격 : ", y_pred[1])

y1_true = np.array(range(3101,3107)) # 온도 
y2_true = np.array(range(13101,13107))

r2_1 = r2_score(y1_true, y_pred[0])
r2_2 = r2_score(y2_true, y_pred[1])
print("r2_1 : ", r2_1, "r2_2 : ", r2_2 )

rmse_1 = np.sqrt(mean_squared_error(y1_true, y_pred[0]))
rmse_2 = np.sqrt(mean_squared_error(y2_true, y_pred[1]))
print("rmse_1 : ", rmse_1, "rmse_2 : ", rmse_2, )

# 결과
# loss : [2.558001756668091, 2.518453359603882, 0.039548300206661224]
# 온도 : 
# [[3090.523 ]
#  [3092.611 ]
#  [3094.7014]
#  [3096.789 ]
#  [3098.8777]
#  [3100.9666]] 
# 가격 :  
# [[13068.072]
#  [13074.793]
#  [13081.513]
#  [13088.23 ]
#  [13094.949]
#  [13101.669]]
# r2_1 :  -20.80587387084961 r2_2 :  -150.69241333007812

# 결과 2
# loss : [553.7749633789062, 525.9631958007812, 27.811771392822266]
# 온도 : 
# [[3058.7676]
#  [3061.9033]
#  [3065.0454]
#  [3068.3672]
#  [3071.7192]
#  [3075.0713]] 
# 가격 :  
# [[13085.225]
#  [13098.817]
#  [13112.422]
#  [13126.686]
#  [13141.065]
#  [13155.443]]
# r2_1 :  -465.6150817871094 r2_2 :  -262.3005065917969
# rmse_1 :  36.89119910343201 rmse_2 :  27.71208721536563


