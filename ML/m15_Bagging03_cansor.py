import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier


#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=888,
    stratify=y,
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

model = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=100, #epochs
    random_state=422,
    n_jobs=-1,
    max_samples=0.7,# 샘플의 비율 정할 수 있음
    bootstrap=True, # 디폴트 # 중복허용 
    # bootstrap=False, # 중복 안됨(불가)
)


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result =  model.score(x_test, y_test)
print('최종 점수 :', result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc_score', acc)

# model = BaggingClassifier(
#     RandomForestClassifier(),
#     n_estimators=50, #epochs
#     random_state=48282,
#     n_jobs=-1,
#     max_samples=0.8,# 샘플의 비율 정할 수 있음
#     bootstrap=False, # 중복 안됨(불가)
# 최종 점수 : 0.956140350877193
# acc_score 0.956140350877193

# model = RandomForestClassifier()
# 최종 점수 : 0.956140350877193
# acc_score 0.956140350877193

# model = DecisionTreeClassifier()
# 최종 점수 : 0.9210526315789473
# acc_score 0.9210526315789473

# model = BaggingClassifier(
#     DecisionTreeClassifier(),
#     n_estimators=50, #epochs
#     random_state=48282,
#     n_jobs=-1,
#     max_samples=0.8,# 샘플의 비율 정할 수 있음
#     bootstrap=True, # 디폴트 # 중복허용 
#     # bootstrap=False, # 중복 안됨(불가)
# )
# 최종 점수 : 0.956140350877193
# acc_score 0.956140350877193

# model = BaggingClassifier(
#     DecisionTreeClassifier(),
#     n_estimators=100, #epochs
#     random_state=422,
#     n_jobs=-1,
#     max_samples=0.7,# 샘플의 비율 정할 수 있음
#     bootstrap=True, # 디폴트 # 중복허용 
#     # bootstrap=False, # 중복 안됨(불가)
# )
# 최종 점수 : 0.9736842105263158
# acc_score 0.9736842105263158

# 한가지 모델을 데이터만 가지고 요리조리 돌려본다 (k폴드 연상)
