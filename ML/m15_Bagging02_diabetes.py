import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor


#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=896,
    # stratify=y,
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()

model = BaggingRegressor(
    RandomForestRegressor(),
    n_estimators=50, #epochs
    random_state=48282,
    n_jobs=-1,
    max_samples=0.8,# 샘플의 비율 정할 수 있음
    # bootstrap=True, # 디폴트 # 중복허용 
    bootstrap=False, # 중복 안됨(불가)
)


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result =  model.score(x_test, y_test)
print('최종 점수 :', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2_score', r2)


# model = RandomForestRegressor()
# 최종 점수 : 0.4931519231870637
# r2_score 0.4931519231870637

# model = BaggingRegressor(
#     RandomForestRegressor(),
#     n_estimators=50, #epochs
#     random_state=38282,
#     n_jobs=-1,
#     max_samples=0.8,# 샘플의 비율 정할 수 있음
#     bootstrap=True, # 디폴트 # 중복허용 
# )
# 최종 점수 : 0.525539359082347
# r2_score 0.525539359082347


# model = BaggingRegressor(
#     RandomForestRegressor(),
#     n_estimators=50, #epochs
#     random_state=38282,
#     n_jobs=-1,
#     max_samples=0.8,# 샘플의 비율 정할 수 있음
#     bootstrap=False, # 중복 안됨(불가)
# )
# 최종 점수 : 0.5136858234385707
# r2_score 0.5136858234385707

# model = DecisionTreeRegressor()

# 최종 점수 : 0.05768212769948389
# r2_score 0.05768212769948389



