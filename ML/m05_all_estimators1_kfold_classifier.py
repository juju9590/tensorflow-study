import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict #교차검증
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
import sklearn as sk

import warnings
warnings.filterwarnings('ignore') #워닝있는건 무시하고 넘어가라

#01. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(
    x, y, shuffle=True, random_state=333,train_size=0.8,
    stratify=y, )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#k-fold
n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333) #클래스

#02. 모델
all = all_estimators(type_filter='classifier')

print('all models :', all)
print('sk version :', sk.__version__) #sk version : 1.9.1
print('모델의 개수 :', len(all)) #모델의 개수 : 44

for name, model in all:
    # print(name)
    try: #예외처리
        model = model()

        #03 훈련, 평가
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, 'cross_val_score 의 정답률 :', scores)
    except:
        print(name, '모델오류')


# models : [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), ('BaggingClassifier', <class 'sklearn.ensemble._bagging.BaggingClassifier'>), ('BernoulliNB', <class 'sklearn.naive_bayes.BernoulliNB'>), ('CalibratedClassifierCV', <class 'sklearn.calibration.CalibratedClassifierCV'>), ('CategoricalNB', <class 'sklearn.naive_bayes.CategoricalNB'>), ('ClassifierChain', <class 'sklearn.multioutput.ClassifierChain'>), ('ComplementNB', <class 'sklearn.naive_bayes.ComplementNB'>), ('DecisionTreeClassifier', <class 'sklearn.tree._classes.DecisionTreeClassifier'>), ('DummyClassifier', <class 'sklearn.dummy.DummyClassifier'>), ('ExtraTreeClassifier', <class 'sklearn.tree._classes.ExtraTreeClassifier'>), ('ExtraTreesClassifier', <class 'sklearn.ensemble._forest.ExtraTreesClassifier'>), ('FixedThresholdClassifier', <class 'sklearn.model_selection._classification_threshold.FixedThresholdClassifier'>), ('GaussianNB', <class 'sklearn.naive_bayes.GaussianNB'>), ('GaussianProcessClassifier', <class 'sklearn.gaussian_process._gpc.GaussianProcessClassifier'>), ('GradientBoostingClassifier', <class 'sklearn.ensemble._gb.GradientBoostingClassifier'>), ('HistGradientBoostingClassifier', <class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier'>), ('KNeighborsClassifier', <class 'sklearn.neighbors._classification.KNeighborsClassifier'>), ('LabelPropagation', <class 'sklearn.semi_supervised._label_propagation.LabelPropagation'>), ('LabelSpreading', <class 'sklearn.semi_supervised._label_propagation.LabelSpreading'>), ('LinearDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'>), ('LinearSVC', <class 'sklearn.svm._classes.LinearSVC'>), ('LogisticRegression', <class 'sklearn.linear_model._logistic.LogisticRegression'>), ('LogisticRegressionCV', <class 'sklearn.linear_model._logistic.LogisticRegressionCV'>), ('MLPClassifier', <class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>), ('MultiOutputClassifier', <class 'sklearn.multioutput.MultiOutputClassifier'>), ('MultinomialNB', <class 'sklearn.naive_bayes.MultinomialNB'>), ('NearestCentroid', <class 'sklearn.neighbors._nearest_centroid.NearestCentroid'>), ('NuSVC', <class 'sklearn.svm._classes.NuSVC'>), ('OneVsOneClassifier', <class 'sklearn.multiclass.OneVsOneClassifier'>), ('OneVsRestClassifier', <class 'sklearn.multiclass.OneVsRestClassifier'>), ('OutputCodeClassifier', <class 'sklearn.multiclass.OutputCodeClassifier'>), ('PassiveAggressiveClassifier', <class 'sklearn.linear_model._passive_aggressive.PassiveAggressiveClassifier'>), ('Perceptron', <class 'sklearn.linear_model._perceptron.Perceptron'>), ('QuadraticDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'>), ('RadiusNeighborsClassifier', <class 'sklearn.neighbors._classification.RadiusNeighborsClassifier'>), ('RandomForestClassifier', <class 'sklearn.ensemble._forest.RandomForestClassifier'>), ('RidgeClassifier', <class 'sklearn.linear_model._ridge.RidgeClassifier'>), ('RidgeClassifierCV', <class 'sklearn.linear_model._ridge.RidgeClassifierCV'>), ('SGDClassifier', <class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>), ('SVC', <class 'sklearn.svm._classes.SVC'>), ('SelfTrainingClassifier', <class 'sklearn.semi_supervised._self_training.SelfTrainingClassifier'>), ('StackingClassifier', <class 'sklearn.ensemble._stacking.StackingClassifier'>), ('TunedThresholdClassifierCV', <class 'sklearn.model_selection._classification_threshold.TunedThresholdClassifierCV'>), ('VotingClassifier', <class 'sklearn.ensemble._voting.VotingClassifier'>)]
# sk version : 1.9.1
# 모델의 개수 : 44
# AdaBoostClassifier cross_val_score 의 정답률 : [0.91666667 0.91666667 0.95833333 1.         0.91666667]
# BaggingClassifier cross_val_score 의 정답률 : [0.95833333 0.95833333 0.95833333 1.         0.95833333]
# BernoulliNB cross_val_score 의 정답률 : [0.66666667 0.75       0.70833333 0.75       0.95833333]
# CalibratedClassifierCV cross_val_score 의 정답률 : [0.83333333 0.875      0.875      0.95833333 0.875     ]
# CategoricalNB 모델오류
# ClassifierChain 모델오류
# ComplementNB 모델오류
# DecisionTreeClassifier cross_val_score 의 정답률 : [1.         0.95833333 0.95833333 1.         0.95833333]
# DummyClassifier cross_val_score 의 정답률 : [0.20833333 0.20833333 0.29166667 0.20833333 0.20833333]
# ExtraTreeClassifier cross_val_score 의 정답률 : [0.95833333 0.95833333 0.875      1.         1.        ]
# ExtraTreesClassifier cross_val_score 의 정답률 : [0.95833333 0.95833333 0.91666667 1.         0.95833333]
# FixedThresholdClassifier 모델오류
# GaussianNB cross_val_score 의 정답률 : [0.875      0.95833333 0.95833333 1.         0.95833333]
# GaussianProcessClassifier cross_val_score 의 정답률 : [0.91666667 0.95833333 0.91666667 1.         0.95833333]
# GradientBoostingClassifier cross_val_score 의 정답률 : [0.95833333 0.95833333 0.95833333 1.         0.95833333]
# HistGradientBoostingClassifier cross_val_score 의 정답률 : [0.91666667 0.91666667 0.95833333 1.         0.91666667]
# KNeighborsClassifier cross_val_score 의 정답률 : [1.         0.95833333 0.83333333 1.         0.95833333]
# LabelPropagation cross_val_score 의 정답률 : [0.91666667 0.95833333 0.875      1.         0.95833333]
# LabelSpreading cross_val_score 의 정답률 : [0.91666667 0.95833333 0.875      1.         0.95833333]
# LinearDiscriminantAnalysis cross_val_score 의 정답률 : [0.95833333 1.         0.95833333 1.         0.95833333]
# LinearSVC cross_val_score 의 정답률 : [0.91666667 1.         0.95833333 1.         0.91666667]
# LogisticRegression cross_val_score 의 정답률 : [0.95833333 0.95833333 0.91666667 1.         0.95833333]
# LogisticRegressionCV cross_val_score 의 정답률 : [0.95833333 1.         0.91666667 1.         0.95833333]
# MLPClassifier cross_val_score 의 정답률 : [0.875      0.95833333 0.91666667 1.         0.95833333]
# MultiOutputClassifier 모델오류
# MultinomialNB 모델오류
# NearestCentroid cross_val_score 의 정답률 : [0.75       0.83333333 0.91666667 1.         0.83333333]
# NuSVC cross_val_score 의 정답률 : [0.95833333 0.95833333 0.91666667 1.         0.95833333]
# OneVsOneClassifier 모델오류
# OneVsRestClassifier 모델오류
# OutputCodeClassifier 모델오류
# PassiveAggressiveClassifier cross_val_score 의 정답률 : [0.79166667 0.95833333 0.83333333 0.91666667 0.95833333]
# Perceptron cross_val_score 의 정답률 : [0.875      0.79166667 0.83333333 0.875      0.75      ]
# QuadraticDiscriminantAnalysis cross_val_score 의 정답률 : [0.95833333 1.         0.83333333 1.         1.        ]
# RadiusNeighborsClassifier cross_val_score 의 정답률 : [0.91666667 0.91666667        nan 1.         0.91666667]
# RandomForestClassifier cross_val_score 의 정답률 : [0.95833333 0.95833333 0.95833333 1.         0.91666667]
# RidgeClassifier cross_val_score 의 정답률 : [0.79166667 0.83333333 0.83333333 0.91666667 0.79166667]
# RidgeClassifierCV cross_val_score 의 정답률 : [0.79166667 0.83333333 0.83333333 0.91666667 0.79166667]
# SGDClassifier cross_val_score 의 정답률 : [0.83333333 1.         0.95833333 0.91666667 0.95833333]
# SVC cross_val_score 의 정답률 : [0.95833333 1.         0.875      1.         0.95833333]
# SelfTrainingClassifier 모델오류
# StackingClassifier 모델오류
# TunedThresholdClassifierCV 모델오류
# VotingClassifier 모델오류


