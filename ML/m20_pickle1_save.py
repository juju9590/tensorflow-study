# 19-1 copy

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBRegressor, XGBClassifier

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=999,
    # stratify=y,
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters ={'n_estimators' : 1500,
             'learning_rate':0.1,
             'max_depth': 9,
             'gamma':1,
             'min_child_weight':1,
             'subsample':1,
             'colsample_bytree': 1,
             'colsample_bylevel': 1,
             'colsample_bynode': 1,
             'reg_alpha':0,
             'reg_lambda':1,
             'ramdom_state': 888,
            #  'verbose':0,             
             }

# *args : 여러 개의 값을 [[튜플]]로 받기
# **kwargs : 이름이 있는 값을 [[딕셔너리]]로 받기

#2. 모델
model = XGBRegressor(
    # **parameters, # 딕셔너리 가져올때 앞에 붙이기 (변수자리에)
)

model.set_params(**parameters,
                 early_stopping_rounds=20,
                 )

#3. 훈련
model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = 1,
          )

#4. 평가, 예측
result =  model.score(x_test, y_test)
print('최종 점수 :', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2_score', r2)

# 스태킹
# CatBoostRegressor ACC : 0.8464 (기준)

# model = XGBRegressor() 디폴트
# 최종 점수 : 0.8393594171205966
# r2_score 0.8393594171205966


# 최종 점수 : 0.8270992733087654
# r2_score 0.8270992733087654

path ='./_save/pickle/'
# import joblib 
import pickle

# joblib.dump(model, path +'m19_joblib1_save.joblib')
pickle.dump(model, open(path +'m20_pickle1_save.pickle', 'wb'))

# 최종 점수 : 0.8270992849969493
# r2_score 0.8270992849969493
