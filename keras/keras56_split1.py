# 시계열 timesteps 따라 데이터 나누기

import numpy as np

a = np.array(range(1,11))
size = 5                # timestep 사이즈 

print(a.shape)           # (10,) 벡터


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
'''
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]
 '''
print(bbb.shape) #(6, 5) => batch=6, timestep=5, feature=1 (1열)
print(len(a)) #10

# subset
print(a[0:5]) #[1 2 3 4 5] ======== 0
print(a[1:6]) #[2 3 4 5 6] ======== 1
print(a[2:7]) #[3 4 5 6 7] ======== 2
print(a[3:8]) #[4 5 6 7 8] ======== 3
print(a[4:9]) #[5 6 7 8 9] ======== 4
print(a[5:10]) #[ 6  7  8  9 10] == 5

# x, y 데이터 분리 (리스트 Split)
# bbb[행, 열]
# 1)
# x_data = bbb[:, :4] # 모든 행, 인덱스 0부터 3까지 (4열 선택)
# y_data = bbb[:, 4]  # 모든 행, 인데스 4열 선택

# print("x_data", x_data)
# print("y_data", y_data)

# 2)
print(bbb.shape[1])
x_data = bbb[:, :(bbb.shape[1]-1)] # 모든 행, 인덱스 0부터 3까지 (4열 선택)
y_data = bbb[:, (bbb.shape[1]-1)]  # 모든 행, 인데스 4열 선택

print("x_data", x_data)
print("y_data", y_data)












