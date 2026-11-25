
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
    x, y, train_size=0.8, random_state=3444,
    # stratify=y,
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2-1. 모델
xgb = XGBRFRegressor()
lg = LGBMRegressor(verbosity=-1)
cat = CatBoostRegressor(verbose=0)
rf = RandomForestRegressor()

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

    r2 = r2_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print('{0} ACC : {1:.4f}'.format(class_name, r2))

# XGBRFRegressor ACC : 0.6946
# LGBMRegressor ACC : 0.8307
# CatBoostRegressor ACC : 0.8464
# RandomForestRegressor ACC : 0.8037

x_train_new = np.array(train_list).T
print(x_train_new.shape) #(16512, 4)

x_test_new = np.array(test_list).T
print(x_test_new.shape) #(4128, 4)

#2-2 모델
# model2 = CatBoostRegressor(verbose=0)
# model2 = XGBRFRegressor()
model2 = RandomForestRegressor()

#3-2. 훈련
model2.fit(x_train_new, y_train)

#4. 평가 예측
y_pred_new = model2.predict(x_test_new)
r2_new = r2_score(y_test, y_pred_new)
print('스태킹 결과:', r2_new )


# XGBRFRegressor ACC : 0.4194
# LGBMRegressor ACC : 0.3420
# CatBoostRegressor ACC : 0.4172
# RandomForestRegressor ACC : 0.4263
# (353, 4)
# (89, 4)
# 
# model2 = CatBoostRegressor(verbose=0)
# 스태킹 결과: 0.4142629263280856

#===================================

# XGBRFRegressor ACC : 0.4194
# LGBMRegressor ACC : 0.3420
# CatBoostRegressor ACC : 0.4172
# RandomForestRegressor ACC : 0.4186
# (353, 4)
# (89, 4)

# model2 = RandomForestRegressor()
# 스태킹 결과: 0.4090796386619554








