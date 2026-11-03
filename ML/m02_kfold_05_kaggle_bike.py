# 42-5 카피

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차검증
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
path = "./_data/kaggle_bike/" # 학원
# path = "D:\\Furiosa_AI\\study_3주차\\_data\\kaggle_bike\\" # 개인

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

########### x, y 분리 #############3

x = train_csv.drop(['casual', 'registered', 'count'], axis=1) #열(컬럼) 삭제 
# print(x) # [10886 rows x 8 columns]

y = train_csv['count']
# print(y) # (10886,)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=142,
    )


# K-Fold (스케일링 필요 없음 )
n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=12345) #클래스


# 2. 모델 구성
model = RandomForestRegressor()


# 3-4. 훈련, 평가
scores = cross_val_score(model, x, y, cv=kfold)
print('R2 : ', scores)
print('평균 R2 :', round(np.mean(scores),3) )


# k-fold 결과
# R2 :  [0.27403433 0.27320141 0.33989305 0.30734487 0.24035184]
# 평균 R2 : 0.287

####### dnn >>>> cnn 1차
# r2 :  0.32492876052856445