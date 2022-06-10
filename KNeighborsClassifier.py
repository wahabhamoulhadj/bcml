import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import glob
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

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


parameter_grid = {'n_neighbors': list(range(1, 15)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
model = KNeighborsClassifier()
main(model, parameter_grid)


'''
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 13}
Accuracy : 0.7952083333333333
Test Accuracy: 0.38720538720538716
Tuned Hyper parameters : {'algorithm': 'brute', 'n_neighbors': 2}
Accuracy : 0.6146895015316068
Test Accuracy: 0.5
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 12}
Accuracy : 0.7663963963963963
Test Accuracy: 0.5014064697609001
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 14}
Accuracy : 0.7994686994686995
Test Accuracy: 0.38580645161290317
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 8}
Accuracy : 0.6922222222222223
Test Accuracy: 0.9900990099009901
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 14}
Accuracy : 0.6210839578581513
Test Accuracy: 0.5613095238095237
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 14}
Accuracy : 0.684107796493935
Test Accuracy: 0.4678688909774436
Tuned Hyper parameters : {'algorithm': 'brute', 'n_neighbors': 12}
Accuracy : 0.6280146328146328
Test Accuracy: 0.5199115044247788
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 8}
Accuracy : 0.7489761928249646
Test Accuracy: 0.43067617866004965
Tuned Hyper parameters : {'algorithm': 'brute', 'n_neighbors': 7}
Accuracy : 0.7146834015247047
Test Accuracy: 0.5552447552447554
Tuned Hyper parameters : {'algorithm': 'brute', 'n_neighbors': 13}
Accuracy : 0.8227728628313423
Test Accuracy: 0.5675675675675675
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 13}
Accuracy : 0.8322259136212624
Test Accuracy: 0.31500572737686144
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 10}
Accuracy : 0.8105109890109891
Test Accuracy: 0.4752155172413793
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 12}
Accuracy : 0.8375852272727273
Test Accuracy: 0.33501006036217307
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 12}
Accuracy : 0.7499813258636788
Test Accuracy: 0.4560357675111774
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 13}
Accuracy : 0.8272438672438673
Test Accuracy: 0.521585557299843
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 10}
Accuracy : 0.791044776119403
Test Accuracy: 0.5218390804597701
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 14}
Accuracy : 0.8473260073260074
Test Accuracy: 0.6617647058823529
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 9}
Accuracy : 0.7044046789736444
Test Accuracy: 0.44319131161236425
Tuned Hyper parameters : {'algorithm': 'brute', 'n_neighbors': 14}
Accuracy : 0.678595414462081
Test Accuracy: 0.47683836912996
Tuned Hyper parameters : {'algorithm': 'brute', 'n_neighbors': 3}
Accuracy : 0.6855929487179487
Test Accuracy: 0.56875
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 8}
Accuracy : 0.7824561403508772
Test Accuracy: 0.2608695652173913
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 9}
Accuracy : 0.7420674132630655
Test Accuracy: 0.4155701754385965
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 11}
Accuracy : 0.7618055555555555
Test Accuracy: 0.41946064139941686
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 14}
Accuracy : 0.6735010822510823
Test Accuracy: 0.4831349206349206
Tuned Hyper parameters : {'algorithm': 'brute', 'n_neighbors': 14}
Accuracy : 0.7182332373508845
Test Accuracy: 0.560044642857143
Tuned Hyper parameters : {'algorithm': 'auto', 'n_neighbors': 9}
Accuracy : 0.7692465535861762
Test Accuracy: 0.35969868173258

'''