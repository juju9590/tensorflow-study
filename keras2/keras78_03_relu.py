import numpy as np
import matplotlib.pyplot as plt

x = np. arange(-5, 5, 0.1)

# def relu(x):                   # 0보다 작으면 0, 0보다 크면 x
#     return np.maximum(0, x)

relu = lambda x : np.maximum(0, x)

y= relu(x)
plt.plot(x, y)
plt.grid()
plt.show() 




