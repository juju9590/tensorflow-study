import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict #교차검증
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


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

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=999999) #클래스

#02. 모델
model = DecisionTreeClassifier()

#03-4. 훈련, 평가
# 컴파일 없음
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc : ', scores,'\n평균 acc :', round(np.mean(scores),3) )


y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)
print(y_test)

acc_score = accuracy_score(y_test, y_predict)
print('acc_score : ', acc_score)


#==================================================================
# acc :  [1.         1.         0.95833333 0.95833333 0.91666667] 
# 평균 acc : 0.967
# [2 0 2 1 2 1 1 0 2 0 0 0 0 2 2 2 0 2 0 1 1 0 1 1 2 0 1 1 1 1]
# [2 0 2 1 1 1 2 0 2 0 0 0 0 2 2 2 0 2 0 1 2 0 1 1 2 0 1 1 1 1]
# acc_score :  0.9
# =================================================================
# acc :  [1.         0.93333333 0.93333333 0.96666667 0.93333333] 
# ㄴ 각 데이터별 훈련의 가중치는 합쳐지지 않는다 
# 평균 acc : 0.953
# ㄴ 각 훈련 결과의 평균
