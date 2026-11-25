# 투표방식 (예시 : 3개의 모델에서 2개이상 같으면 선택, 다수결)
# soft voting vs hard voting

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
    x, y, train_size=0.9, random_state=3444,
    stratify=y,
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBClassifier()
lg = LGBMClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(
    estimators=[('xgb', xgb), ('lg',lg), ('rf', rf), ('cat', cat)],
    # voting='hard' # 디폴트
    voting ='soft',
 )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result =  model.score(x_test, y_test)
print('최종 점수 :', result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc_score', round(acc,3))


# model = VotingClassifier(
#     estimators=[('xgb', xgb), ('lg',lg), ('rf', rf), ('cat', cat)],
#     voting='hard' # 디폴트
#     # voting ='soft',
#  )
# 최종 점수 : 0.9824561403508771
# acc_score 0.982


# model = VotingClassifier(
#     estimators=[('xgb', xgb), ('lg',lg), ('rf', rf), ('cat', cat)],
#     # voting='hard' # 디폴트
#     voting ='soft',
#  )
# 최종 점수 : 0.9824561403508771
# acc_score 0.982

