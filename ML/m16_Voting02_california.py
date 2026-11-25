# 투표방식 (예시 : 3개의 모델에서 2개이상 같으면 선택, 다수결)
# soft voting vs hard voting

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRFRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=444,
    # stratify=y,
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBRFRegressor()
lg = LGBMRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor()

model = VotingRegressor(
    estimators=[('xgb', xgb), ('lg',lg), ('rf', rf), ('cat', cat)],
 )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result =  model.score(x_test, y_test)
print('최종 점수 :', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2_score', r2)

# model = RandomForestRegressor() ==> 알고리즘은 배깅
# 최종 점수 : 0.8114223340789829
# r2_score 0.8114223340789829

# ======================

# model = VotingRegressor(
#     estimators=[('xgb', xgb), ('lg',lg), ('rf', rf), ('cat', cat)],
#  )
# train_size=0.8, random_state=9888,

# 최종 점수 : 0.8331698048342363
# r2_score 0.8331698048342363