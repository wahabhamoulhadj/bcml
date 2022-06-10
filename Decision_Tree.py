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


parameter_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'max_features': ['sqrt', 'log2'],
                  'criterion': ['gini', 'entropy', 'log_loss']}
model = DecisionTreeClassifier()
main(model, parameter_grid)



'''
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 2, 'max_features': 'sqrt'}
Accuracy : 0.8216666666666667
Test Accuracy: 0.6397306397306397
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 8, 'max_features': 'log2'}
Accuracy : 0.6779587858535228
Test Accuracy: 0.5
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 2, 'max_features': 'sqrt'}
Accuracy : 0.788933933933934
Test Accuracy: 0.5
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 3, 'max_features': 'log2'}
Accuracy : 0.7808847308847309
Test Accuracy: 0.5
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 2, 'max_features': 'log2'}
Accuracy : 0.72
Test Accuracy: 0.44554455445544555
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 7, 'max_features': 'sqrt'}
Accuracy : 0.6089644835451288
Test Accuracy: 0.5416666666666667
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 3, 'max_features': 'log2'}
Accuracy : 0.6885727572757275
Test Accuracy: 0.53125
Tuned Hyper parameters : {'criterion': 'gini', 'max_depth': 8, 'max_features': 'log2'}
Accuracy : 0.6524680589680589
Test Accuracy: 0.5199115044247787
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'log2'}
Accuracy : 0.7652556640689225
Test Accuracy: 0.5066687344913151
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 3, 'max_features': 'log2'}
Accuracy : 0.7145489087868588
Test Accuracy: 0.48721278721278727
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 3, 'max_features': 'sqrt'}
Accuracy : 0.8242190233418304
Test Accuracy: 0.5438635356668144
Tuned Hyper parameters : {'criterion': 'gini', 'max_depth': 3, 'max_features': 'sqrt'}
Accuracy : 0.7707641196013288
Test Accuracy: 0.4215349369988546
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 3, 'max_features': 'log2'}
Accuracy : 0.7999130036630037
Test Accuracy: 0.5043103448275862
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 3, 'max_features': 'log2'}
Accuracy : 0.7717045454545455
Test Accuracy: 0.5
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 7, 'max_features': 'sqrt'}
Accuracy : 0.683361344537815
Test Accuracy: 0.5692995529061103
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 5, 'max_features': 'sqrt'}
Accuracy : 0.8349350649350649
Test Accuracy: 0.5
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 3, 'max_features': 'log2'}
Accuracy : 0.8716856892010536
Test Accuracy: 0.5
Tuned Hyper parameters : {'criterion': 'gini', 'max_depth': 3, 'max_features': 'log2'}
Accuracy : 0.8186446886446885
Test Accuracy: 0.4579831932773109
Tuned Hyper parameters : {'criterion': 'gini', 'max_depth': 4, 'max_features': 'log2'}
Accuracy : 0.70522030651341
Test Accuracy: 0.48955722639933164
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt'}
Accuracy : 0.6965562610229277
Test Accuracy: 0.5113608785341585
Tuned Hyper parameters : {'criterion': 'gini', 'max_depth': 11, 'max_features': 'log2'}
Accuracy : 0.6963301282051282
Test Accuracy: 0.5125
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 3, 'max_features': 'sqrt'}
Accuracy : 0.7991228070175438
Test Accuracy: 0.5
Tuned Hyper parameters : {'criterion': 'gini', 'max_depth': 3, 'max_features': 'sqrt'}
Accuracy : 0.7232268335529205
Test Accuracy: 0.5224780701754386
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 2, 'max_features': 'sqrt'}
Accuracy : 0.7535353535353535
Test Accuracy: 0.5102040816326531
Tuned Hyper parameters : {'criterion': 'gini', 'max_depth': 3, 'max_features': 'log2'}
Accuracy : 0.7203409090909092
Test Accuracy: 0.5828373015873016
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 6, 'max_features': 'sqrt'}
Accuracy : 0.7210287261757851
Test Accuracy: 0.45
Tuned Hyper parameters : {'criterion': 'gini', 'max_depth': 3, 'max_features': 'log2'}
Accuracy : 0.7888434661076171
Test Accuracy: 0.5018832391713748
'''