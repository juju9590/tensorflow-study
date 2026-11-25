import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor


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
# model = DecisionTreeRegressor()
model = RandomForestRegressor()

# model = BaggingRegressor(
#     DecisionTreeRegressor(),
#     n_estimators=50, #epochs
#     random_state=38282,
#     n_jobs=-1,
#     max_samples=0.8,# 샘플의 비율 정할 수 있음
#     # bootstrap=True, # 디폴트 # 중복허용 
#     bootstrap=False, # 중복 안됨(불가)
# )
# 앙상블은 모델을 섞어서 만들지만, 회귀는 하나만 쓴다.
# 랜덤포레스트가 bagging 기법


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result =  model.score(x_test, y_test)
print('최종 점수 :', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2_score', r2)


# model = DecisionTreeRegressor()
# 최종 점수 : 0.5726668897489695
# r2_score 0.572666889748969


# model = BaggingRegressor(
#     DecisionTreeRegressor(),
#     n_estimators=10, #epochs
#     random_state=333,
#     n_jobs=-1,
#     max_samples=0.8,# 샘플의 비율 정할 수 있음
#   ***  bootstrap=True, # 디폴트 # 중복허용 
# )
# 최종 점수 : 0.7813565822670153
# r2_score 0.7813565822670153

# model = BaggingRegressor(
#     DecisionTreeRegressor(),
#     n_estimators=10, #epochs
#     random_state=333,
#     n_jobs=-1,
#     max_samples=0.8,# 샘플의 비율 정할 수 있음
#    *** bootstrap=False, # 중복 안됨(불가)
# )
# 최종 점수 : 0.7735648120632528
# r2_score 0.7735648120632528

# 배깅 , bootstrap=False, 랜덤 999
# 최종 점수 : 0.7910397815852686
# r2_score 0.7910397815852686

# 배깅 , bootstrap=False, 랜덤 38282, n_estimators=50,
# 최종 점수 : 0.8046394889648766
# r2_score 0.8046394889648766

# model = RandomForestRegressor() ==> 알고리즘은 배깅
# 최종 점수 : 0.8114223340789829
# r2_score 0.8114223340789829




