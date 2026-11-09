# ML/m09_RandomSearchCV_00.py 카피
# Halving = 이등분, 반땡 ==> 큰 데이터에 강력, 작은 데이터에는 효용성 떨어짐 

import numpy as np
from sklearn.datasets import load_digits, load_iris, load_diabetes
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection._search_successive_halving import HalvingGridSearchCV

from xgboost import XGBClassifier
import time

import warnings
warnings.filterwarnings('ignore')

#01. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test,y_train,y_test =train_test_split(
    x,y, shuffle=True, random_state=333,test_size=0.2,
    stratify=y,
)
print(x_train.shape, y_train.shape)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = [
    {"n_estimators": [100,500], 'max_depth' : [6,10,12],
     'learning_rate': [0.1,0.01,0.001]}, 

     {'max_depth': [6,8,10,12],'learning_rate': [0.1,0.01,0.001] }, 
     {'min_child_weight': [2,3,5,10], 'learning_rate': [0.1,0.01,0.001] }, 
]    

#2. 모델
xgb = XGBClassifier()
model = HalvingGridSearchCV(xgb, parameters, cv=kfold,  
             verbose=1,
            #  n_jobs=-1, 
            #  random_state=333,
             refit=True, 
            #  n_iter=10, 
            max_resources=1200, # x_train의 1437행 
            min_resources=80, # 2*cv*클래스갯수
            factor=4,
            )

# 03. 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print("최적의 매개변수 :", model.best_estimator_) 
print("최적의 파라미터 :", model.best_params_)

#04. 평가, 예측
print("===============================================")
print('model.socore :',model.score(x_test, y_test))
print('best_score :', model.best_score_) 

print('걸린시간 :', round(end-start,2),'초')

# (1437, 64) (1437,)
# n_iterations: 3
# n_required_iterations: 4
# n_possible_iterations: 3
# min_resources_: 100
# max_resources_: 1437
# aggressive_elimination: False
# factor: 3 # 디폴트
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 100
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 14
# n_resources: 300
# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# ----------
# iter: 2
# n_candidates: 5
# n_resources: 900
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               feature_weights=None, gamma=None, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
#               max_leaves=None, min_child_weight=2, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=None,
#               n_jobs=None, num_parallel_tree=None, ...)
# 최적의 파라미터 : {'learning_rate': 0.1, 'min_child_weight': 2}
# model.socore : 0.9444444444444444
# best_score : 0.9431160769708257
# 걸린시간 : 57.2 초


# foctor=2로 파라미터 설정시 
# (1437, 64) (1437,)
# n_iterations: 4
# n_required_iterations: 6
# n_possible_iterations: 4
# min_resources_: 100
# max_resources_: 1437
# aggressive_elimination: False
# factor: 2
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 100
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 21
# n_resources: 200
# Fitting 5 folds for each of 21 candidates, totalling 105 fits
# ----------
# iter: 2
# n_candidates: 11
# n_resources: 400
# Fitting 5 folds for each of 11 candidates, totalling 55 fits
# ----------
# iter: 3
# n_candidates: 6
# n_resources: 800
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               feature_weights=None, gamma=None, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=12,
#               max_leaves=None, min_child_weight=None, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=500,
#               n_jobs=None, num_parallel_tree=None, ...)
# 최적의 파라미터 : {'learning_rate': 0.1, 'max_depth': 12, 'n_estimators': 500}
# model.socore : 0.9444444444444444
# best_score : 0.9385298742138366
# 걸린시간 : 66.05 초


# (1437, 64) (1437,)
# n_iterations: 4
# n_required_iterations: 6
# n_possible_iterations: 4
# min_resources_: 100
# max_resources_: 1437
# aggressive_elimination: False
# factor: 2
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 100
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 21
# n_resources: 200
# Fitting 5 folds for each of 21 candidates, totalling 105 fits
# ----------
# iter: 2
# n_candidates: 11
# n_resources: 400
# Fitting 5 folds for each of 11 candidates, totalling 55 fits
# ----------
# iter: 3
# n_candidates: 6
# n_resources: 800
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               feature_weights=None, gamma=None, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=12,
#               max_leaves=None, min_child_weight=None, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=500,
#               n_jobs=None, num_parallel_tree=None, ...)
# 최적의 파라미터 : {'learning_rate': 0.1, 'max_depth': 12, 'n_estimators': 500}
# model.socore : 0.9444444444444444
# best_score : 0.9385298742138366
# 걸린시간 : 66.05 초