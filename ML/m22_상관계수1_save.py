import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



#1. 데이터
# datasets = load_iris()
# print(datasets.feature_names)
# # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# x = datasets['data'] # 딕셔너리 형태 데이터 불러오기
# y = datasets.target # 클래스 기반 데이터 불러오기

# df = pd.DataFrame(x, columns=datasets.feature_names)
# df['target'] = y
# print(df)

# exit()

#####################################
import joblib 
# joblib 저장

# path ='./_save/joblib/'
# joblib.dump(df, path +'m22_joblib111_save.joblib')

# 불러오기
# df = joblib.load(path +'m22_joblib111_save.joblib')
# print(df)

##################################### 
import pickle
# pickle 저장

path ='./_save/pickle/'
# pickle.dump(df, open(path + 'm22_pickle111_save.pickle','wb'))

# 불러오기
# df = pickle.load(open(path + 'm22_pickle111_save.pickle','rb'))
# print(df)


# 상관관계 : 1 높다, 0 관계없다, 절대값으로 판단




