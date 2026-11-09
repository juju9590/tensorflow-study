# ML/m08_GridSearchCV_00.py 카피

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# RandomizedSearchCV : 그리드서치는 전체를 다 돌리지만, 랜덤을 
from xgboost import XGBClassifier
import time

import warnings
warnings.filterwarnings('ignore')

#01. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test,y_train,y_test =train_test_split(
    x,y, shuffle=True, random_state=333,test_size=0.8,
    stratify=y,
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = [
    {"n_estimators": [100,500], 'max_depth' : [6,10,12],
     'learning_rate': [0.1,0.01,0.001]}, #18번

     {'max_depth': [6,8,10,12],'learning_rate': [0.1,0.01,0.001] }, # 12번
     {'min_child_weight': [2,3,5,10], 'learning_rate': [0.1,0.01,0.001] }, #12
]    # 총 42

#2. 모델
xgb = XGBClassifier()
model = RandomizedSearchCV(xgb, parameters, cv=kfold,  # 210번 돌아감(42*5)
        # 하이퍼파라미터를 랜덤으로 뽑겠다 (위 42개중 10개를 뽑아준다)
        # 랜덤으로 뽑기때문에 고정되지 않는다, 파라미터 중 random_state=333로 고정시켜주면 
             verbose=1,
             n_jobs=-1, #cpu 다 돌리기
             random_state=333,
             refit=True, #사실은 최적의 파라미터로 1번 더 돌아가기 때문에  51번 실행
            #  refit=False, # 최적의 파라미터를 알 수 없기때문에 오류남
             
             n_iter=3, # 총 42개 중에 몇개 선택할지 정하는거
            #  n_iter는 hyperparameters 중에 몇 개 고를지 정하는 int값이라 
            # 디폴트인 10이나 다른 수로 변경해주면 좋을 것 같습니다
            )

# 03. 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print("최적의 매개변수 :", model.best_estimator_) #estimator=평가자
print("최적의 파라미터 :", model.best_params_)

#04. 평가, 예측
print('model.socore :',model.score(x_test, y_test))
print('best_score :', model.best_score_) #train 데이터에 대한 스코어 (과적합이 발생함)

print('걸린시간 :', round(end-start,2),'초')

# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               feature_weights=None, gamma=None, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.01, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=12,
#               max_leaves=None, min_child_weight=None, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=None,
#               n_jobs=None, num_parallel_tree=None, ...)
# 최적의 파라미터 : {'max_depth': 12, 'learning_rate': 0.01}
# model.socore : 0.8916666666666667
# best_score : 0.9333333333333333
# 걸린시간 : 5.25 초


# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               feature_weights=None, gamma=None, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.01, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
#               max_leaves=None, min_child_weight=2, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=None,
#               n_jobs=None, num_parallel_tree=None, ...)
# 최적의 파라미터 : {'min_child_weight': 2, 'learning_rate': 0.01}
# model.socore : 0.8916666666666667
# best_score : 0.9333333333333333
# 걸린시간 : 5.41 초


 


