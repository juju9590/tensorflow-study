import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
import tensorflow as tf
tf.random.set_seed(333)
np.random.seed(333)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#1. 데이터
x_data = np.array([[0,0],[0,1],[1,0],[1,1]]) #(4,2)
y_data = np.array([0,1,1,0]) #(4,)
# xor = 같으면 0, 다르면 1

#2. 모델
# model = Perceptron()
# model = LinearSVC()
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

# model = Sequential()
# model.add(Dense(10, input_dim=2, activation=None)) #다층모델 구현
# model.add(BatchNormalization())
# model.add(Dense(1, activation='sigmoid'))
# model.add(Activation('relu'))

#3. 컴파일, 훈련 
# model.compile(loss='binary_crossentropy', 
#               optimizer='adam',
#               metrics=['acc'],)

# model.fit(x_data, y_data)

# #4. 평가 
# y_predict = model.predict(x_data)
# result = model.evaluate(x_data, y_data)
# print("acc: ", result[1])

# acc = accuracy_score(y_data, np.round(y_predict))
# print("acc_accuracy :", acc)

#3. 컴파일, 훈련
model.fit(x_data, y_data)
print("====================",model.__class__.__name__,"===================")

#4. 평가, 성능
score = model.score(x_data, y_data)  # model.evaluate(x_test, y_test)
print("acc : ", score)
print(model.feature_importances_)

# 다층 퍼셉트론(히든O)
# acc:  0.75
# acc_accuracy : 0.75

# activation='elu'
# acc:  1.0
# acc_accuracy : 1.0

# activation=None
# acc:  0.5
# acc_accuracy : 0.5



