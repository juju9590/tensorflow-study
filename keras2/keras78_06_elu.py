import numpy as np
import matplotlib.pyplot as plt

x = np. arange(-5, 5, 0.1)

alpha = 1

# def elu(x, alpha):                   
#     return (x>0)*x +(x<0)*(alpha*(np.exp(x)-1)) 

# x가 양수 : (x>0)*x
# x가 음수 : (x<0)*(alpha*(np.exp(x)-1)) 

elu = lambda x, alpha : (x>0)*x +(x<0)*(alpha*(np.exp(x)-1)) 

y= elu(x, alpha)  # alpha=파라미터, x가 음수일때 최저값을 지정한다.
plt.plot(x, y)
plt.grid()
plt.show() 

# elu는 음수를 조정할 수 있다. 
# 이것도 하이퍼 파라미터 
# 디폴트 = 1