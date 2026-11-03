import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict #교차검증
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
import sklearn as sk

import warnings
warnings.filterwarnings('ignore') #워닝있는건 무시하고 넘어가라

#01. 데이터
x, y = load_iris(return_X_y=True)
# print(x,y)
# print(x.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test=train_test_split(
    x, y, shuffle=True, random_state=333,train_size=0.8,
    stratify=y,
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=999999) #클래스

#02. 모델
all = all_estimators(type_filter='classifier')

print('all models :', all)
print('sk version :', sk.__version__) #sk version : 1.9.1
print('모델의 개수 :', len(all)) #모델의 개수 : 44

# all = all_estimators(type_filter='regressor')

# print('all models :', all)
# print('sk version :', sk.__version__) #sk version : 1.9.1
# print('모델의 개수 :', len(all)) #모델의 개수 : 55

for name, model in all:
    # print(name)
    try: #예외처리
        model = model()

        #03 훈련, 
        model.fit(x_train, y_train)

        #04. 평가
        acc = model.score(x_test, y_test)
        print(name, '의 정답률 :', acc)
    except:
        print(name, '모델오류')


