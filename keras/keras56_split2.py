import numpy as np

a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0],
              ]).T
print(a.shape) #(10, 2)

size = 4

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)

print(bbb)
'''
[[[ 1  9]
  [ 2  8]
  [ 3  7]
  [ 4  6]]

 [[ 2  8]
  [ 3  7]
  [ 4  6]
  [ 5  5]]

 [[ 3  7]
  [ 4  6]
  [ 5  5]
  [ 6  4]]

 [[ 4  6]
  [ 5  5]
  [ 6  4]
  [ 7  3]]

 [[ 5  5]
  [ 6  4]
  [ 7  3]
  [ 8  2]]

 [[ 6  4]
  [ 7  3]
  [ 8  2]
  [ 9  1]]

 [[ 7  3]
  [ 8  2]
  [ 9  1]
  [10  0]]]
'''
print(bbb.shape) # (7, 4, 2) => RNN 3차 (batch=7, timestep=4, feature=2)2열
print("batch" ,bbb.shape[0])
print("timestep", bbb.shape[1])
print("feature",bbb.shape[2])


# x_data = bbb[:, :3]
x_data = bbb[:, :(bbb.shape[1]-1)]

print("x_data:", x_data)

# y_data = bbb[:, 3]
y_data = bbb[:, (bbb.shape[1]-1)]

print("y_data:", y_data)
