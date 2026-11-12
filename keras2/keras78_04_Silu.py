# 2017년 구글에서 swish 발표
# x에 sigmoid() 곱한 함수 : silu = swish 같은거.. 기억하기
# sigmoid liner Unit

import numpy as np
import matplotlib.pyplot as plt

x = np. arange(-5, 5, 0.1)

def silu(x):                   
    return x * (1/(1+np.exp(-x)))  # 0이하를 0이 아니라 값을  넣어줬더니 성능이 좋을때도 있더라 
                                   # relu의 음수쪽을 개선한 

# exit()

silu = lambda x : x * (1/(1+np.exp(-x)))  #람다 매개변수(X) : return 값

y= silu(x) 
plt.plot(x, y)
plt.grid()
plt.show() 



