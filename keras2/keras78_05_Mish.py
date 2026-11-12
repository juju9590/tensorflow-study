import numpy as np
import matplotlib.pyplot as plt

x = np. arange(-5, 5, 0.1)

# def mish(x):                   
#     return x * np.tanh(np.log(1+np.exp(x)))  
# 연산량이 Silu보다 많다
# 그래프는 Silu 와 모양이 비슷하게 나옴. 연산량이 크면 silu 를 사용하면 된다.

mish = lambda x : x * np.tanh(np.log(1+np.exp(x)))

y= mish(x) 
plt.plot(x, y)
plt.grid()
plt.show() 