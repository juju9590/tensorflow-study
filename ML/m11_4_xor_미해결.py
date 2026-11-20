import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
import tensorflow as tf
tf.random.set_seed(333)
np.random.seed(333)

#1. 데이터
x_data = np.array([[0,0],[0,1],[1,0],[1,1]]) #(4,2)
y_data = np.array([0,1,1,0]) #(4,)
# xor = 같으면 0, 다르면 1

#2. 모델
# model = Perceptron()
# model = LinearSVC()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid')) #단층모델 구현
model.add(BatchNormalization())

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['acc'],)

model.fit(x_data, y_data, epochs=1000)

#4. 평가 
y_predict = model.predict(x_data)
result = model.evaluate(x_data, y_data)
print("acc: ", result[1])

acc = accuracy_score(y_data, np.round(y_predict))
print("acc_accuracy :", acc)


# 단층 퍼셉트론(히든이 없다)
# loss, acc:  [0.7111042737960815, 0.5]
# acc_accuracy : 0.5


# acc:  0.75
# acc_accuracy : 0.75

