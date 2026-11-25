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

# imputer = SimpleImputer(strategy='most_frequent') #최빈값
# data2 = imputer.fit_transform(data)
# print(data2)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# 5   8.0  4.0  12.0  8.0
# [[ 2.  2.  2.  8.]
#  [ 8.  4.  4.  4.]
#  [ 6.  4.  6.  8.]
#  [ 8.  8.  8.  8.]
#  [10.  4. 10.  8.]
#  [ 8.  4. 12.  8.]]

# imputer2 = SimpleImputer(strategy='constant', fill_value=777) #특정값은 constant, 값을 지정해 줘야함
# data3 = imputer2.fit_transform(data)
# print(data3)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# 5   8.0  4.0  12.0  8.0
# [[  2.   2.   2. 777.]
#  [777.   4.   4.   4.]
#  [  6. 777.   6. 777.]
#  [  8.   8.   8.   8.]
#  [ 10. 777.  10. 777.]
#  [  8.   4.  12.   8.]]

# def 함수():
#     return 1

imputer3 = SimpleImputer(
    # strategy=lambda x : np.max(x)
    strategy=lambda x : np.min(x)
)
data4 = imputer3.fit_transform(data)
print(data4)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# 5   8.0  4.0  12.0  8.0
# [[ 2.  2.  2.  4.]
#  [ 2.  4.  4.  4.]
#  [ 6.  2.  6.  4.]
#  [ 8.  8.  8.  8.]
#  [10.  2. 10.  4.]
#  [ 8.  4. 12.  8.]]
