import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

#01. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test,y_train,y_test =train_test_split(
    x,y, shuffle=True, random_state=333,test_size=0.8,
    stratify=y,
)

learnig_rate = [0.1, 0.05, 0.01, 0.005, 0.001]
max_depth = [3,4,5,6,7]
best_score = 0
best_parameters = ''

for i, lr in enumerate(learnig_rate): #enumerate = 리스트를 감싸서 인덱스와 벨류로 빼주는 기능
# i = 0부터 4까지
    for j, md in enumerate(max_depth):
        model=HistGradientBoostingClassifier(
            learning_rate=lr,
            max_depth=md
            # max_depth = 결정트리가 최대 몇 단계까지 질문을 이어갈 수 있는지 정하는 값, 
            # 즉 트리의 최대 깊이예요.
        )
        model.fit(x_train,y_train)
        score = model.score(x_test,y_test)

        if score > best_score:
            best_score = score
            best_parameters = lr, md
        print(i, ",", j, "번째 score :", round(score,4), "실행중...")
print("최고점수 :{:.2f}".format(best_score)) #
print("최적 매개변수 :", best_parameters)

# 최고점수 :0.95
# 최적 매개변수 : (0.1, 6)