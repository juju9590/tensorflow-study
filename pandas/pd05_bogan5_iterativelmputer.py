import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data = pd.DataFrame([[2, np.nan, 6,8,10, 8],
                      [2, 4, np.nan, 8, np.nan, 4],
                      [2, 4, 6, 8, 10, 12],
                      [np.nan, 4, np.nan, 8,np.nan, 8],
                      ]).T

data.columns = ['x1','x2','x3','x4']
print(data)

from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=100,    # 순차적으로 학습할 트리 개수(epochs, 디폴트 100번)
    max_depth=5,         # 트리의 최대 깊이
    learning_rate=0.1,   # 새 트리의 반영 정도
    tol=0.001, # 디폴트 (얼리스탑의 기준)
    random_state=333
)
# 과적합 걸릴수 있으니 잘 알아보고 하기 

imputer = IterativeImputer(
    estimator=xgb,
    max_iter=10,
    random_state=333,
)

data2 = imputer.fit_transform(data)
print(data2)
print("실제 반복 횟수:", imputer.n_iter_)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# 5   8.0  4.0  12.0  8.0
# C:\Users\Admin\miniconda3\envs\ml-311\Lib\site-packages\xgboost\training.py:200: UserWarning: [13:44:02] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:782: 
# Parameters: { "tol" } are not used.

#   bst.update(dtrain, iteration=i, fobj=obj)
# [[ 2.          2.          2.          7.99865484]
#  [ 9.98037434  4.          4.          4.        ]
#  [ 6.          2.01480126  6.          7.99865484]
#  [ 8.          8.          8.          8.        ]
#  [10.          7.97927809 10.          4.01578808]
#  [ 8.          4.         12.          8.        ]]
# 실제 반복 횟수: 3