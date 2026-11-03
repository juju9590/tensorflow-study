import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차검증
from sklearn.ensemble import RandomForestRegressor


#01. 데이터
datasets = fetch_california_housing() 
x = datasets.data
y = datasets.target

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=999999) #클래스


#02. 모델
model = RandomForestRegressor()

#03-4. 훈련, 평가
# 컴파일 없음
scores = cross_val_score(model, x, y, cv=kfold)
print('R2 : ', scores)
print('평균 R2 :', round(np.mean(scores),3) )

# R2 :  [0.81205631 0.80616974 0.806022   0.80573527 0.81997357]
# 평균 R2 : 0.81

