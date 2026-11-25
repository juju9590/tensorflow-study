import pandas as pd
import numpy as np


data = pd.DataFrame([[2, np.nan, 6,8,10],
                      [2, 4, np.nan, 8, np.nan],
                      [2, 4, 6, 8, 10],
                      [np.nan, 4, np.nan, 8,np.nan],
                      ]).T

data.columns = ['x1','x2','x3','x4']
print(data)
# #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   x1      4 non-null      float64
#  1   x2      3 non-null      float64
#  2   x3      5 non-null      float64
#  3   x4      2 non-null      float64

from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer

imputer = SimpleImputer() # 디폴트 = 평균값 (결측치를 평균값으로 채워진다)
# data2 = imputer.fit_transform(data)
# print(data2)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]

imputer2 = SimpleImputer(strategy='mean')
data3 = imputer2.fit_transform(data)
print(data3)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]

# imputer3 = SimpleImputer(strategy='median')
# data4 = imputer3.fit_transform(data)
# print(data4)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# [[ 2.  2.  2.  6.]
#  [ 7.  4.  4.  4.]
#  [ 6.  4.  6.  6.]
#  [ 8.  8.  8.  8.]
#  [10.  4. 10.  6.]]

