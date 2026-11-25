# 15-1 copy

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBRegressor, XGBClassifier
import joblib 


#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=999,
    # stratify=y,
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
path ='./_save/joblib/'
model = joblib.load(path +'m19_joblib1_save.joblib')

#3. 훈련

#4. 평가, 예측
result =  model.score(x_test, y_test)
print('최종 점수 :', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2_score', r2)

# (기준)
# 최종 점수 : 0.8270992849969493
# r2_score 0.8270992849969493

#==============================
# 불러오기
# 최종 점수 : 0.8270992849969493
# r2_score 0.8270992849969493



