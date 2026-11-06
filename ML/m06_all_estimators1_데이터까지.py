import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer #데이터
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold 
# StratifiedKFold 분류모델에서 사용, train_test_split의 stratify=y와 같은 개념
from sklearn.model_selection import cross_val_score, cross_val_predict #교차검증
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
import sklearn as sk
import time

import warnings
warnings.filterwarnings('ignore') #워닝있는건 무시하고 넘어가라

#01. 데이터
iris = load_iris(return_X_y=True)
# print(iris)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

datasets = [iris, cancer, wine, digits]
data_name = ['아이리스','캔서','와인', '디지트']

#02. 모델
all = all_estimators(type_filter='classifier')
# all = all_estimators(type_filter='regressor')

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333) 

start = time.time()
for index, value in enumerate(datasets):
    x, y = value # 데이터를 x, y로 구분 [iris, cancer, wine, digits]

    x_train, x_test, y_train, y_test=train_test_split(
        x, y, shuffle=True, random_state=333,train_size=0.8,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("============", data_name[index],"============")
    model_name=[]
    model_acc=[]

    for name, model in all:
        # print(name)
        try: #예외처리
            model = model()

            #03 훈련, 
            scores = cross_val_score(model, x_train, y_train, cv=kfold) #cv=5 는 kfold 디폴트 값으로 적용

            #04. 평가
            y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

            acc = np.round(accuracy_score(y_test, y_pred),4)
            model_name.append(name)
            model_acc.append(acc)

        except:
            # print(name, '모델오류')
            continue

    # model_name = np.array(model_name)
    model_acc = np.array(model_acc)
    max_index = np.argmax(model_acc)
    print('최고 모델 :', model_name[max_index], 'ACC :', model_acc[max_index], 'model_index :', max_index)

end = time.time()
print("걸린시간 : ", round(end-start,2),'초')

# ============ 아이리스 ============
# 최고 모델 : ExtraTreesClassifier ACC : 0.9667
# ============ 캔서 ============
# 최고 모델 : AdaBoostClassifier ACC : 0.9825
# ============ 와인 ============
# 최고 모델 : ExtraTreesClassifier ACC : 1.0
# ============ 디지트 ============
# 최고 모델 : ExtraTree
# 걸린시간 :  84.15 초

# ============ 아이리스 ============
# 최고 모델 : ExtraTreesClassifier ACC : 0.9667 model_index : 7
# ============ 캔서 ============
# 최고 모델 : AdaBoostClassifier ACC : 0.9825 model_index : 0
# ============ 와인 ============
# 최고 모델 : ExtraTreesClassifier ACC : 1.0 model_index : 7
# ============ 디지트 ============
# 최고 모델 : ExtraTreesClassifier ACC : 0.95 model_index : 7
# 걸린시간 :  85.98 초