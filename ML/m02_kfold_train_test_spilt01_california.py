import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


#01. 데이터
datasets = fetch_california_housing() 
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=142,
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=999999) #클래스


#02. 모델
model = RandomForestRegressor()

#03-4. 훈련, 평가
# 컴파일 없음
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('R2 : ', scores)
print('평균 R2 :', round(np.mean(scores),3) )

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)
print(y_test)

r2 = r2_score(y_test, y_predict)
print("cross_val_predict_r2 : ", r2)

# =============================================================
# R2 :  [0.80910333 0.80892903 0.7761707  0.80327476 0.8140562 ]
# 평균 R2 : 0.802
# cross_val_predict_r2 :  0.7525403508300457

#===============================================================
# R2 :  [0.81205631 0.80616974 0.806022   0.80573527 0.81997357]
# 평균 R2 : 0.81

