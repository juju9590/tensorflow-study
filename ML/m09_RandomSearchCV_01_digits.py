#디지트

import numpy as np
from sklearn.datasets import load_digits, load_iris, load_diabetes
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# RandomizedSearchCV : 그리드 서치를 
from xgboost import XGBClassifier
import time

import warnings
warnings.filterwarnings('ignore')

#01. 데이터
x, y = load_digits(return_X_y=True)

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
             verbose=1,
             n_jobs=-1,
             random_state=11111,
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

# RandomizedSearchCV 결과
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               feature_weights=None, gamma=None, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=12,
#               max_leaves=None, min_child_weight=None, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=100,
#               n_jobs=None, num_parallel_tree=None, ...)
# 최적의 파라미터 : {'n_estimators': 100, 'max_depth': 12, 'learning_rate': 0.1}
# model.socore : 0.9179415855354659
# best_score : 0.869131455399061



# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               feature_weights=None, gamma=None, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=12,
#               max_leaves=None, min_child_weight=None, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=None,
#               n_jobs=None, num_parallel_tree=None, ...)
# 최적의 파라미터 : {'max_depth': 12, 'learning_rate': 0.1}
# model.socore : 0.9179415855354659
# best_score : 0.869131455399061
# 걸린시간 : 6.89 초
