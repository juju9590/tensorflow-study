#'캘리포니아','디아벳'

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes #데이터
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold 
# StratifiedKFold 분류모델에서 사용, train_test_split의 stratify=y와 같은 개념
from sklearn.model_selection import cross_val_score, cross_val_predict #교차검증
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from sklearn.utils import all_estimators
import sklearn as sk
import time

import warnings
warnings.filterwarnings('ignore') #워닝있는건 무시하고 넘어가라

#01. 데이터
california = fetch_california_housing(return_X_y=True)
# print(iris)
diabetes = load_diabetes(return_X_y=True)


datasets = [diabetes, california]
data_name = ['diavetes','califonia']

#02. 모델
# all = all_estimators(type_filter='classifier')
all = all_estimators(type_filter='regressor')

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333) 

start = time.time()
for index, value in enumerate(datasets):
    x, y = value # 데이터를 x, y로 구분 [iris, cancer, wine, digits]

    x_train, x_test, y_train, y_test=train_test_split(
        x, y, shuffle=True, random_state=333,train_size=0.8,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("============", data_name[index],"============")
    model_name=[]
    model_R2=[]

    for name, model in all:
        # print(name)
        try: #예외처리
            model = model()

            #03 훈련, 
            scores = cross_val_score(model, x_train, y_train, cv=kfold) #cv=5 는 kfold 디폴트 값으로 적용

            #04. 평가
            y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

            scores = np.round(r2_score(y_test, y_pred),4)
            # Fold 5개의 평균 R²
            R2 = scores.mean()  

            model_name.append(name)
            model_R2.append(R2)

        except:
            # print(name, '모델오류')
            continue

    # model_name = np.array(model_name)
    model_R2 = np.array(model_R2)
    max_index = np.argmax(model_R2)
    print('최고 모델 :', model_name[max_index], 'R2 :', model_R2[max_index], 'model_index :', max_index)

end = time.time()
print("걸린시간 : ", round(end-start,2),'초')

# ============ 캘리포니아 ============
# 최고 모델 : HistGradientBoostingRegressor R2 : 0.7947 model_index : 12
# ============ 디아벳 ============
# 최고 모델 : GammaRegressor R2 : 0.4407 model_index : 10
# 걸린시간 :  1190.24 초


# ============ diavetes ============
# 최고 모델 : GammaRegressor R2 : 0.4407 model_index : 10
# ============ califonia ============
# 최고 모델 : HistGradientBoostingRegressor R2 : 0.7947 model_index : 12
# 걸린시간 :  3862.16 초