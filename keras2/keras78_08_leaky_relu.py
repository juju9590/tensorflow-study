import numpy as np
import matplotlib.pyplot as plt

x = np. arange(-5, 5, 0.1)

alpha = 0.01

# def leaky_relu(x):                   
#     # return np.maximum(alpha*x, x)
#     return np.where(x>0, x, alpha*x) # x>0 ==> x, x<0 ==> alpha*x

# leaky_relu = lambda x : np.maximum(alpha*x, x)
leaky_relu = lambda x : np.where(x>0, x, alpha*x)

y= leaky_relu(x) 
plt.plot(x, y)
plt.grid()
plt.show() 