import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import joblib 


#1. 데이터
path ='./_save/joblib/'

df = joblib.load(path +'m22_joblib111_save.joblib')
print(df)

print("======== 상관관계 히트맵 ========")
print(df.corr())


# 상관관계 : 1 (-1) 높다, 0 관계없다, 절대값으로 판단
# X 끼리는 상관이 높으면 중복으로 생각하고 X와 Y와 상관이 높으면 주요 특성으로 본다.
# 상관관계는 단순 리니어 이기 때문에, 이거 한가지 결과만 보고 판단 하지 말것
# 피처 인폴턴스와 함께 결합해서 의미를 찾아야 함

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()

