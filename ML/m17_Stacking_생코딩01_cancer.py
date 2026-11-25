# Pseudo Labeling 기법 : 모델을 돌려서 나온 결과로 결측치를 찾음
# 스태킹 : 모델을 돌려 나온거로 컬럼을 구성해서 새로운 데이터셋을 만듦
#       : 한 데이터로 여러 모델을 돌려서 돌리는 족족 컬럼을 만듦

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRFRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=3444,
    stratify=y,
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2-1. 모델
xgb = XGBClassifier()
lg = LGBMClassifier(verbosity=-1)
cat = CatBoostClassifier(verbose=0)
rf = RandomForestClassifier()

models = [xgb, lg, cat, rf, ]
train_list = []
test_list = []

for model in models:
    #3-1. 훈련
    model.fit(x_train, y_train)

    y_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_list.append(y_pred)
    test_list.append(y_test_pred)

    score = accuracy_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print('{0} ACC : {1:.4f}'.format(class_name, score))

# x, y, train_size=0.9, random_state=3444,
# XGBClassifier ACC : 0.9825
# LGBMClassifier ACC : 0.9737
# CatBoostClassifier ACC : 0.9912
# RandomForestClassifier ACC : 0.9561

# x_train_new = np.array(train_list)
# print(x_train_new.shape) # (4, 455)

x_train_new = np.array(train_list).T
print(x_train_new.shape) #(455, 4)

x_test_new = np.array(test_list).T
print(x_test_new.shape) #(114, 4)

#2-2 모델
# model2 = CatBoostClassifier(verbose=0)
model2 = XGBClassifier()

#3-2. 훈련
model2.fit(x_train_new, y_train)

#4. 평가 예측
y_pred_new = model2.predict(x_test_new)
score_new = accuracy_score(y_test, y_pred_new)
print('스태킹 결과:', score_new )

# model2 = CatBoostClassifier(verbose=0)
# 스태킹 결과: 0.9912280701754386

# model2 = XGBClassifier()
# 스태킹 결과: 0.9824561403508771





