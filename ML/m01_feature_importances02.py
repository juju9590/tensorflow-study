from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1. 데이터
x, y = load_diabetes(return_X_y=True) #x,y 분리
print(x.shape, y.shape) #

random_state = 9999

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=random_state, 
    # stratify=y, # 회귀에서 쓸수 없다. y의 비율대로 맞춰라
)

#2. 모델구성
model1 = DecisionTreeRegressor(random_state=random_state)  
model2 = RandomForestRegressor(random_state=random_state) 
model3 = GradientBoostingRegressor(random_state=random_state) 
model4 = XGBRegressor(random_state=random_state) 

models = [model1, model2, model3, model4]

for model in models:

    #3. 컴파일, 훈련
    model.fit(x_train, y_train)
    print("====================",model.__class__.__name__,"===================")

    #4. 평가, 성능
    score = model.score(x_test, y_test)  # model.evaluate(x_test, y_test)
    print("R2 : ", score)
    print(model.feature_importances_) #컬럼의 중요도를 볼 수 있다. 


# (442, 10) (442,)
# ==================== DecisionTreeRegressor ===================
# R2 :  0.06617636458105824
# [0.0668302  0.0127425  0.1680079  0.17135689 0.05940238 0.0493693
#  0.02050748 0.01939866 0.37092458 0.06146011]
# ==================== RandomForestRegressor ===================
# R2 :  0.4282982337473573
# [0.0689965  0.00882465 0.19997928 0.12487889 0.0518495  0.05239618
#  0.05390908 0.02409031 0.35963682 0.05543878]
# ==================== GradientBoostingRegressor ===================
# R2 :  0.42316647071405566
# [0.05796959 0.01753765 0.20068037 0.12718998 0.06256664 0.04688985
#  0.05313097 0.00982002 0.3962301  0.02798483]
# ==================== XGBRegressor ===================
# R2 :  0.24118368242177657
# [0.03547102 0.06329884 0.1294336  0.13499445 0.06096017 0.05826385
#  0.06154865 0.0710642  0.33902934 0.04593595]





