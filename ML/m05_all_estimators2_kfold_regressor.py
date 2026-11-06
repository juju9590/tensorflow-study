import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict #교차검증
from sklearn.preprocessing import StandardScaler

from sklearn.utils import all_estimators
import sklearn as sk

import warnings
warnings.filterwarnings('ignore') #워닝있는건 무시하고 넘어가라

#01. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(
    x, y, shuffle=True, random_state=333,train_size=0.8,
    stratify=y,
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#k-fold
n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333) #클래스


#02. 모델
all = all_estimators(type_filter='regressor')

print('all models :', all)
print('sk version :', sk.__version__) #sk version : 1.9.1
print('모델의 개수 :', len(all)) #모델의 개수 : 55


for name, model in all:
    try:
        model = model()

        #03. 훈련, 평가
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, 'r2 정답율 :', scores)
    except:
        print(name, '오류')

# ARDRegression r2 정답율 : [0.90983521 0.91330442 0.88830853 0.96407488 0.92512785]
# AdaBoostRegressor r2 정답율 : [0.94438563 0.91540175 0.93684211 1.         0.94139194]
# BaggingRegressor r2 정답율 : [0.94225    0.91524752 0.94252632 1.         0.96043956]
# BayesianRidge r2 정답율 : [0.91220453 0.91457664 0.8920984  0.96401405 0.92189888]
# CCA 오류
# DecisionTreeRegressor r2 정답율 : [0.925      0.84158416 0.93684211 1.         0.94725275]
# DummyRegressor r2 정답율 : [-0.078125   -0.04641089 -0.01644737 -0.27173913 -0.00343407]
# ElasticNet r2 정답율 : [0.43290755 0.44923608 0.44836275 0.2838868  0.39849063]
# ElasticNetCV r2 정답율 : [0.90680678 0.91460002 0.89066674 0.96373394 0.92220446]
# ExtraTreeRegressor r2 정답율 : [0.925      1.         0.81052632 0.93478261 0.94725275]
# ExtraTreesRegressor r2 정답율 : [0.9624175  0.96312871 0.93793474 0.99823261 0.96274989]
# GammaRegressor 오류
# GaussianProcessRegressor r2 정답율 : [ 0.80190117  0.7853785   0.77828393 -0.69565217  0.85878481]
# GradientBoostingRegressor r2 정답율 : [0.93213736 0.88724348 0.93684553 0.99978285 0.96225639]
# HistGradientBoostingRegressor r2 정답율 : [0.95963974 0.89053202 0.92996689 0.99793922 0.94183402]
# HuberRegressor r2 정답율 : [0.91242887 0.91704105 0.89734408 0.96317939 0.91153904]
# IsotonicRegression 오류
# KNeighborsRegressor r2 정답율 : [0.967      0.90811881 0.856      0.9973913  0.94303297]
# KernelRidge r2 정답율 : [-0.96445114 -1.07567791 -0.60244022 -0.92895066 -0.5326241 ]
# Lars r2 정답율 : [0.9124443  0.91466154 0.89223112 0.96254166 0.92093772]
# LarsCV r2 정답율 : [0.9124443  0.91466154 0.88887264 0.96254166 0.92093772]
# Lasso r2 정답율 : [-0.078125   -0.04641089 -0.01644737 -0.27173913 -0.00343407]
# LassoCV r2 정답율 : [0.91188932 0.91462469 0.88899287 0.96365609 0.92202935]
# LassoLars r2 정답율 : [-0.078125   -0.04641089 -0.01644737 -0.27173913 -0.00343407]
# LassoLarsCV r2 정답율 : [0.9124443  0.91462919 0.88887264 0.96371826 0.92093772]
# LassoLarsIC r2 정답율 : [0.90666683 0.91466154 0.88745397 0.96254166 0.92497334]
# LinearRegression r2 정답율 : [0.9124443  0.91466154 0.89223112 0.96254166 0.92093772]
# LinearSVR r2 정답율 : [0.90977997 0.91325124 0.8968571  0.96261426 0.89069666]
# MLPRegressor r2 정답율 : [0.9193756  0.93353964 0.78329622 0.98027263 0.95536344]
# MultiOutputRegressor 오류
# MultiTaskElasticNet 오류
# MultiTaskElasticNetCV 오류
# MultiTaskLasso 오류
# MultiTaskLassoCV 오류
# NuSVR r2 정답율 : [0.94046068 0.94013083 0.87533622 0.98397142 0.93086922]
# OrthogonalMatchingPursuit r2 정답율 : [0.88766477 0.85102691 0.89697654 0.97090981 0.90552509]
# OrthogonalMatchingPursuitCV r2 정답율 : [0.9124443  0.91466154 0.87638333 0.96254166 0.92093772]
# PLSCanonical 오류
# PLSRegression r2 정답율 : [0.9007879  0.89732836 0.86728494 0.96557344 0.91603682]
# PassiveAggressiveRegressor r2 정답율 : [0.60840815 0.80545674 0.67350597 0.68357948 0.9514498 ]
# PoissonRegressor r2 정답율 : [0.69496951 0.65197992 0.67971267 0.62032686 0.68146676]
# QuantileRegressor r2 정답율 : [-0.05       -0.02970297 -0.01052632 -0.17391304 -0.0021978 ]
# RANSACRegressor r2 정답율 : [0.91196455 0.91466154 0.89024781 0.96254166 0.90675273]
# RadiusNeighborsRegressor r2 정답율 : [ 8.94898491e-01  8.36427268e-01 -5.37287948e+36  9.70337724e-01
#   8.99474813e-01]
# RandomForestRegressor r2 정답율 : [0.9513625  0.91678416 0.93905263 0.99841522 0.96537143]
# RegressorChain 오류
# Ridge r2 정답율 : [0.91175507 0.91403471 0.89152904 0.96496991 0.92312172]
# RidgeCV r2 정답율 : [0.91175507 0.91403471 0.89152904 0.96496991 0.92312172]
# SGDRegressor r2 정답율 : [0.88538803 0.85465337 0.868176   0.95289164 0.9358178 ]
# SVR r2 정답율 : [0.93708061 0.93569599 0.86497928 0.97649674 0.92693688]
# StackingRegressor 오류
# TheilSenRegressor r2 정답율 : [0.91127113 0.91892272 0.89200949 0.95731115 0.91226669]
# TransformedTargetRegressor r2 정답율 : [0.9124443  0.91466154 0.89223112 0.96254166 0.92093772]
# TweedieRegressor r2 정답율 : [0.79845857 0.75780942 0.82684411 0.88495344 0.84528441]
# VotingRegressor 오류



