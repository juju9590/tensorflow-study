import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차검증
from sklearn.tree import DecisionTreeClassifier

#01. 데이터
x, y = load_iris(return_X_y=True)
# print(x,y)
# print(x.shape, y.shape) #(150, 4) (150,)




#02. 모델
model = DecisionTreeClassifier()

#03-4. 훈련, 평가
# 컴파일 없음
scores = cross_val_score(model, x, y, cv=kfold)
print('acc : ', scores,'\n평균 acc :', round(np.mean(scores),3) )

# acc :  [1.         0.93333333 0.93333333 0.96666667 0.93333333] 
# ㄴ 각 데이터별 훈련의 가중치는 합쳐지지 않는다 
# 평균 acc : 0.953
# ㄴ 각 훈련 결과의 평균
