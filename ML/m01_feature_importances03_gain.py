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
model = XGBClassifier(random_state=random_state) #1.0


#3. 컴파일, 훈련
model.fit(x_train, y_train)
print("========",model.__class__.__name__,"========")

#4. 평가, 성능
score = model.score(x_test, y_test)  
print("acc : ", score)
print("f-score : ", model.feature_importances_) #컬럼의 중요도 
# f스코어 : 빈도수


gain_score = model.get_booster().get_score(importance_type='gain')
print("gain_feature importances", gain_score)
# Gain 스코어 : 손실함수의 개선 점수, 높으면 개선이 많이 됐다.

# ======== XGBClassifier ========
# acc :  0.9333333333333333
# f-score :  [0.02430454 0.02472077 0.7376847  0.21328993] # 3번째 컬럼을 많이 사용했다 
# gain_feature importances {'f0': 0.09280245006084442, 'f1': 0.09439174085855484, 'f2': 2.8167145252227783, 'f3': 0.8144086599349976}
# ㄴ 손실함수의 개선율 ===>>> 실질적인 성능
# 2가지 스코어가 같은 향상일때는 높은 숫자가 좋은거
# 2가지 스코아가 다르다면, Gain 스코어가 높은것을 선정...왜??? 성능향상에 더 큰 영향을 준다.





