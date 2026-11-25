# 투표방식 (예시 : 3개의 모델에서 2개이상 같으면 선택, 다수결)
# soft voting vs hard voting

import numpy as np
from sklearn.datasets import load_diabetes
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
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.85, random_state=444,
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

# model = BaggingRegressor(
#     RandomForestRegressor(),
# ...
#     bootstrap=True, # 디폴트 # 중복허용 
# )
# 최종 점수 : 0.525539359082347
# r2_score 0.525539359082347

# =====

# model = VotingRegressor(
#     estimators=[('xgb', xgb), ('lg',lg), ('rf', rf), ('cat', cat)],
#  )
# 최종 점수 : 0.5273470686919737
# r2_score 0.5273470686919737
