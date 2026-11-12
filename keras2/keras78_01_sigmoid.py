# 최적의 weight, 최소의 loss
# sigmoid : 0 ~ 1 사이 , 이진분류 

import numpy as np
import matplotlib.pyplot as plt

# def sigmoid(x):                   # 수식 y= 1/(1+e^X)  0 ~ 1 사이값만 가짐, 0이나 1이 될 수 없다
#     return 1/(1 + np.exp(-x))

sigmoid = lambda x : 1/(1 + np.exp(-x)) # 람다함수로 변환 

x = np.arange(-5, 5, 0.1)
print(x)
print(len(x)) #100

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()



