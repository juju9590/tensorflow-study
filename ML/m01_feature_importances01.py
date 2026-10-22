from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
x, y = load_iris(return_X_y=True) #x,y 분리
print(x.shape, y.shape) #(150, 4) (150,)

random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=random_state, 
    stratify=y, # 회귀에서 쓸수 없다. y의 비율대로 맞춰라
)

#2. 모델구성
model1 = DecisionTreeClassifier(random_state=random_state)  #0.8333333333333334
model2 = RandomForestClassifier(random_state=random_state)  #0.9333333333333333
model3 = GradientBoostingClassifier(random_state=random_state) #0.9666666666666667
model4 = XGBClassifier(random_state=random_state) #1.0

models = [model1, model2, model3, model4]

for model in models:

    #3. 컴파일, 훈련
    model.fit(x_train, y_train)
    print("====================",model.__class__.__name__,"===================")

    #4. 평가, 성능
    score = model.score(x_test, y_test)  # model.evaluate(x_test, y_test)
    print("acc : ", score)
    print(model.feature_importances_) #컬럼의 중요도 


# (150, 4) (150,)
# ==================== DecisionTreeClassifier ===================
# acc :  0.8333333333333334
# [0.0125     0.03       0.92133357 0.03616643] #행무시열우선,컬럼=피처=열=어트리뷰트
# ==================== RandomForestClassifier ===================
# acc :  0.9333333333333333
# [0.08860798 0.0214775  0.48716455 0.40274997]
# ==================== GradientBoostingClassifier ===================
# acc :  0.9666666666666667
# [0.00157033 0.02147603 0.82483537 0.15211826]      ==> f.score = frequence.score : 3번째 컬럼을 가장 많이 도달했다고 보면 됨... 
# ==================== XGBClassifier ===================
# acc :  0.9333333333333333
# [0.02430454 0.02472077 0.7376847  0.21328993]

# 회귀에서 model.score 의 결과는 R2


