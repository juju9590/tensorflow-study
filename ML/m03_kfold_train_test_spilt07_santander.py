#42-7


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict 
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


path = 'c:/study/_data/kaggle_santander/' #학원
# path = 'D:\\Furiosa_AI\\study_3주차\\_data\\kaggle_santander\\' #집

train_csv = pd.read_csv(path + "train.csv", index_col=0) #파일이 있는 경로 표시
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)


# 데이터
x = train_csv.drop(['target'], axis=1) 
y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size=0.7,
                 random_state=345,
                 stratify=y,
                 )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# K-Fold (스케일링 필요 없음 )
n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=999999) #클래스

#02. 모델
model = DecisionTreeClassifier()

#03-4. 훈련, 평가
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc : ', scores,'\n평균 acc :', round(np.mean(scores),3) )

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)
print(y_test)

acc_score = accuracy_score(y_test, y_predict)
print('acc_score : ', acc_score)




# =============================================
# k-fold = 5 평가결과
# acc :  [0.8368   0.838075 0.833075 0.836025 0.835625] 
# 평균 acc : 0.836

######### dnn >>> cnn 1차
# acc_score :  0.9217

