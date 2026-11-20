import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]] #(4,2)
y_data = [0,0,0,1] #(4,)

#2. 모델
model = Perceptron()

#3. 훈련 
model.fit(x_data, y_data)

#4. 평가 
y_predict = model.predict(x_data)
result = model.score(x_data, y_data)
print("model_score: ", result)

acc = accuracy_score(y_data, y_predict)
print("acc_accuracy :", acc)


# model_score:  1.0
# acc_accuracy : 1.0