import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import glob
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib
from sklearn.model_selection import cross_val_score

matplotlib.rcParams["figure.figsize"] = (20, 10)


def main(classifier, parameter_grid):
    path = "PROMIS/CK/*.csv"
    for fname in glob.glob(path):
        df1 = pd.read_csv(fname)

        df1.iloc[:, -1:].applymap(lambda x: {'YES': 1, 'NO': 0})
        X = df1.iloc[:, :-1]  # X contains the features
        Y = df1.iloc[:, -1:]  # Y is the target variable

        X = X.to_numpy()
        Y = Y.to_numpy().ravel()

        # Train Test Split Data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=3 / 10, random_state=43)

        # Pre Processing just X_train to avoid Data Leakage
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)

        # Parameter Grid
        clf = GridSearchCV(classifier,  # model
                           param_grid=parameter_grid, refit=True,  # hyperparameters
                           scoring='roc_auc_ovr_weighted',  # metric for scoring
                           cv=5, n_jobs=-1)  # Folds = 5

        clf.fit(X_train, Y_train)  # Training

        print("Tuned Hyper parameters :", clf.best_params_)
        print("Accuracy :", clf.best_score_)
        print("Test Accuracy:", clf.score(X_test, Y_test))


parameter_grid = {'C': np.logspace(-3, 3, 7), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'gamma': ['scale', 'auto']}
model = svm.SVC(probability=True)
main(model, parameter_grid)


'''

Tuned Hyper parameters : {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf'}
Accuracy : 0.8529166666666667
Test Accuracy: 0.5
Tuned Hyper parameters : {'C': 0.01, 'gamma': 'auto', 'kernel': 'sigmoid'}
Accuracy : 0.6164021164021164
Test Accuracy: 0.6385869565217391
Tuned Hyper parameters : {'C': 100.0, 'gamma': 'scale', 'kernel': 'linear'}
Accuracy : 0.8596996996996996
Test Accuracy: 0.44303797468354433
Tuned Hyper parameters : {'C': 10.0, 'gamma': 'auto', 'kernel': 'rbf'}
Accuracy : 0.8114345114345115
Test Accuracy: 0.47333333333333333
Tuned Hyper parameters : {'C': 0.001, 'gamma': 'scale', 'kernel': 'rbf'}
Accuracy : 0.7903703703703704
Test Accuracy: 0.49504950495049505
Tuned Hyper parameters : {'C': 1000.0, 'gamma': 'scale', 'kernel': 'rbf'}
Accuracy : 0.6165634843054197
Test Accuracy: 0.49166666666666664
Tuned Hyper parameters : {'C': 1.0, 'gamma': 'auto', 'kernel': 'poly'}
Accuracy : 0.6961680810938237
Test Accuracy: 0.5111607142857143
Tuned Hyper parameters : {'C': 0.001, 'gamma': 'scale', 'kernel': 'linear'}
Accuracy : 0.6188623805623805
Test Accuracy: 0.5386131084070795
Tuned Hyper parameters : {'C': 1.0, 'gamma': 'auto', 'kernel': 'poly'}
Accuracy : 0.7803995264871263
Test Accuracy: 0.49038461538461536
Tuned Hyper parameters : {'C': 0.001, 'gamma': 'auto', 'kernel': 'sigmoid'}
Accuracy : 0.7252558764027615
Test Accuracy: 0.5012787212787213
Tuned Hyper parameters : {'C': 0.001, 'gamma': 'scale', 'kernel': 'sigmoid'}
Accuracy : 0.8337623498442213
Test Accuracy: 0.2598582188746123
Tuned Hyper parameters : {'C': 0.001, 'gamma': 'scale', 'kernel': 'linear'}
Accuracy : 0.853266888150609
Test Accuracy: 0.4862542955326461
Tuned Hyper parameters : {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
Accuracy : 0.8293150183150182
Test Accuracy: 0.5
Tuned Hyper parameters : {'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf'}
Accuracy : 0.7799431818181819
Test Accuracy: 0.5
Tuned Hyper parameters : {'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf'}
Accuracy : 0.7906535947712419
Test Accuracy: 0.5
Tuned Hyper parameters : {'C': 1000.0, 'gamma': 'auto', 'kernel': 'poly'}
Accuracy : 0.8411544011544011
Test Accuracy: 0.39285714285714285
Tuned Hyper parameters : {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
Accuracy : 0.7615891132572432
Test Accuracy: 0.5
Tuned Hyper parameters : {'C': 100.0, 'gamma': 'auto', 'kernel': 'sigmoid'}
Accuracy : 0.8876923076923078
Test Accuracy: 0.4411764705882352
Tuned Hyper parameters : {'C': 100.0, 'gamma': 'auto', 'kernel': 'poly'}
Accuracy : 0.7372286079182631
Test Accuracy: 0.5079365079365079
Tuned Hyper parameters : {'C': 0.01, 'gamma': 'scale', 'kernel': 'poly'}
Accuracy : 0.684664197530864
Test Accuracy: 0.46726732192695064
Tuned Hyper parameters : {'C': 0.001, 'gamma': 'auto', 'kernel': 'rbf'}
Accuracy : 0.7169871794871795
Test Accuracy: 0.45625
Tuned Hyper parameters : {'C': 0.001, 'gamma': 'auto', 'kernel': 'rbf'}
Accuracy : 0.8122807017543859
Test Accuracy: 0.4673913043478261
Tuned Hyper parameters : {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
Accuracy : 0.7331357048748354
Test Accuracy: 0.5208333333333333
Tuned Hyper parameters : {'C': 10.0, 'gamma': 'auto', 'kernel': 'sigmoid'}
Accuracy : 0.8004873737373737
Test Accuracy: 0.576530612244898
Tuned Hyper parameters : {'C': 0.001, 'gamma': 'scale', 'kernel': 'linear'}
Accuracy : 0.7376298701298701
Test Accuracy: 0.5
Tuned Hyper parameters : {'C': 0.01, 'gamma': 'scale', 'kernel': 'rbf'}
Accuracy : 0.7474592074592075
Test Accuracy: 0.5100446428571429
Tuned Hyper parameters : {'C': 10.0, 'gamma': 'scale', 'kernel': 'linear'}
Accuracy : 0.7845492662473795
Test Accuracy: 0.5550847457627118

'''