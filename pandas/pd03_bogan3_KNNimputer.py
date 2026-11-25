import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer


data = pd.DataFrame([[2, np.nan, 6,8,10, 8],
                      [2, 4, np.nan, 8, np.nan, 4],
                      [2, 4, 6, 8, 10, 12],
                      [np.nan, 4, np.nan, 8,np.nan, 8],
                      ]).T

data.columns = ['x1','x2','x3','x4']
print(data)

# imputer = KNNImputer()
# imputer = KNNImputer(n_neighbors=5) #디폴트
imputer = KNNImputer(n_neighbors=3)
data2 = imputer.fit_transform(data)
print(data2)

# KNN (K-Nearest Neighbors, K-최근접 이웃)
# ㄴ 모델에서도 사용하지만, 결측치, 이상치 처리에도 많이 사용된다 

# imputer = KNNImputer(n_neighbors=5) #디폴트
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# 5   8.0  4.0  12.0  8.0
# [[ 2.          2.          2.          6.66666667]
#  [ 6.8         4.          4.          4.        ]
#  [ 6.          4.5         6.          6.66666667]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.5        10.          6.66666667]
#  [ 8.          4.         12.          8.        ]]

# imputer = KNNImputer(n_neighbors=3)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# 5   8.0  4.0  12.0  8.0
# [[ 2.          2.          2.          6.66666667]
#  [ 5.33333333  4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.66666667]
#  [ 8.          8.          8.          8.        ]
#  [10.          5.33333333 10.          6.66666667]
#  [ 8.          4.         12.          8.        ]]