import numpy as np
aaa = np.array([[-10, 2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T

# aaa = aaa[:,0]
# aaa1 = aaa[:,1]

print(aaa[:,0])
print(aaa[:,1])


def outlier(data1, data2):
    quartile_1, q2, quartile_3 = np.percentile(data1, [25,50,75])
    print('1사분위:', quartile_1)
    print('q2:', q2)
    print('3사분위:', quartile_3)

    quartile_11, q21, quartile_31 = np.percentile(data2, [25,50,75])
    print('1사분위:', quartile_11)
    print('q2:', q21)
    print('3사분위:', quartile_31)

    iqr = quartile_3 - quartile_1
    iqr1 = quartile_31 - quartile_11
    print('IQR :', iqr)
    print('IQR1 :', iqr1)

    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)

    lower_bound1 = quartile_11 - (iqr1 * 1.5)
    upper_bound1 = quartile_31 + (iqr1 * 1.5)

    return np.where((data1 > upper_bound) | (data1 < lower_bound)), \
            iqr, lower_bound, upper_bound, np.where ((data2 > upper_bound1) | (data2 < lower_bound1)), \
            iqr1, lower_bound1, upper_bound1

outlier_loc, iqr, low, up, outlier_loc1, iqr1, low1, up1,  = outlier(aaa[:,0], aaa[:,1])

print('이상치의 위치 :', outlier_loc)
print('이상치의 위치 :', outlier_loc1)

# [-10   2   3   4   5   6   7   8   9  10  11  12  50]
# [   100    200    -30    400    500    600 -70000    800    900   1000
#     210    420    350]

# 1사분위: 4.0
# q2: 7.0
# 3사분위: 10.0

# 1사분위: 200.0
# q2: 400.0
# 3사분위: 600.0

# IQR : 6.0
# IQR1 : 400.0

# 이상치의 위치 : (array([ 0, 12]),)
# 이상치의 위치 : (array([6]),)


# import matplotlib.pyplot as plt
# plt.boxplot(aaa[:,0])
# plt.axhline(up, color='pink', label='upper bound')
# plt.axhline(low, color='pink', label='lower bound')
# plt.legend()
# plt.show()

# import matplotlib.pyplot as plt
# plt.boxplot(aaa[:,1])
# plt.axhline(up1, color='pink', label='upper bound')
# plt.axhline(low1, color='pink', label='lower bound')
# plt.legend()
# plt.show()


# 이상치 처리 1
import pandas as pd
aaa = pd.DataFrame(aaa[:,0]).drop(index=[0,12])
print(aaa)
#      0
# 1    2
# 2    3
# 3    4
# 4    5
# 5    6
# 6    7
# 7    8
# 8    9
# 9   10
# 10  11
# 11  12

aaa1 = pd.DataFrame(aaa[:,1]).drop(index=[6])
print(aaa1)
#        0
# 0    100
# 1    200
# 2    -30
# 3    400
# 4    500
# 5    600
# 7    800
# 8    900
# 9   1000
# 10   210
# 11   420
# 12   350

# # 이상치 처리 2
# aaa_clipped = np.clip(aaa[:,0], low, up)
# print(aaa_clipped)
# # [-5.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 19.]


# aaa1_clipped = np.clip(aaa[:,1], low1, up1)
# print(aaa1_clipped)
# # [ 100.  200.  -30.  400.  500.  600. -400.  800.  900. 1000.  210.  420.
# #   350.]

# 1사분위 아래 이상치 = 1사분위 값으로 대체
# 3사분위 위 이상치 = 3사분위 값으로 대체




