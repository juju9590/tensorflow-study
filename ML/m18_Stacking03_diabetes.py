
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRFRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor

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
xgb = XGBRFRegressor()
lg = LGBMRegressor(verbosity=-1)
cat = CatBoostRegressor(verbose=0)
rf = RandomForestRegressor()

model = StackingRegressor(

    estimators=[('XGB', xgb),('LG',lg),('CAT', cat),('rf',rf)],
    final_estimator=CatBoostRegressor(verbose=0),
    n_jobs=-1,
    cv=5,
)

#3-2. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
y_pred = model.predict(x_test)
score = r2_score(y_test, y_pred)
print('스태킹 결과:', score)

# 이전
# 스태킹 결과: 0.4142629263280856

# 스태킹 결과: 0.9073642353851479