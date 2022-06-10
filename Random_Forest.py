import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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

model = RandomForestClassifier()
main(model, parameter_grid)


'''
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 3, 'max_features': 'log2'}
Accuracy : 0.85875
Test Accuracy: 0.39562289562289576
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 6, 'max_features': 'log2'}
Accuracy : 0.624380395433027
Test Accuracy: 0.529891304347826
Tuned Hyper parameters : {'criterion': 'gini', 'max_depth': 9, 'max_features': 'sqrt'}
Accuracy : 0.818888888888889
Test Accuracy: 0.6765119549929676
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 2, 'max_features': 'sqrt'}
Accuracy : 0.8027835527835527
Test Accuracy: 0.6533333333333333
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 2, 'max_features': 'log2'}
Accuracy : 0.7859259259259259
Test Accuracy: 0.27722772277227725
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 9, 'max_features': 'log2'}
Accuracy : 0.6223851417399805
Test Accuracy: 0.6008597883597885
Tuned Hyper parameters : {'criterion': 'gini', 'max_depth': 3, 'max_features': 'sqrt'}
Accuracy : 0.7098164030688782
Test Accuracy: 0.5837053571428571
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 5, 'max_features': 'log2'}
Accuracy : 0.6909359268359269
Test Accuracy: 0.5344303097345132
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 10, 'max_features': 'sqrt'}
Accuracy : 0.8212776078831123
Test Accuracy: 0.411716811414392
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 7, 'max_features': 'sqrt'}
Accuracy : 0.7690683032595779
Test Accuracy: 0.5063136863136863
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt'}
Accuracy : 0.8426550707252461
Test Accuracy: 0.6074435090828534
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 2, 'max_features': 'sqrt'}
Accuracy : 0.8400332225913623
Test Accuracy: 0.5486827033218785
Tuned Hyper parameters : {'criterion': 'gini', 'max_depth': 2, 'max_features': 'log2'}
Accuracy : 0.8299304029304029
Test Accuracy: 0.47306034482758624
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 6, 'max_features': 'sqrt'}
Accuracy : 0.8143750000000001
Test Accuracy: 0.5368879946344735
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 2, 'max_features': 'sqrt'}
Accuracy : 0.7638095238095237
Test Accuracy: 0.5906607054148039
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 2, 'max_features': 'sqrt'}
Accuracy : 0.8413924963924965
Test Accuracy: 0.5341444270015698
Tuned Hyper parameters : {'criterion': 'gini', 'max_depth': 12, 'max_features': 'sqrt'}
Accuracy : 0.8805970149253731
Test Accuracy: 0.3172413793103448
Tuned Hyper parameters : {'criterion': 'gini', 'max_depth': 2, 'max_features': 'sqrt'}
Accuracy : 0.8612454212454212
Test Accuracy: 0.36344537815126055
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 2, 'max_features': 'sqrt'}
Accuracy : 0.766399628468594
Test Accuracy: 0.5345307713728766
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 6, 'max_features': 'sqrt'}
Accuracy : 0.7561350970017637
Test Accuracy: 0.503291469481859
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2'}
Accuracy : 0.7695192307692307
Test Accuracy: 0.4625
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 2, 'max_features': 'sqrt'}
Accuracy : 0.7447368421052631
Test Accuracy: 0.5652173913043479
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 8, 'max_features': 'sqrt'}
Accuracy : 0.7408212560386473
Test Accuracy: 0.41831140350877194
Tuned Hyper parameters : {'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt'}
Accuracy : 0.8283686868686868
Test Accuracy: 0.48797376093294453
Tuned Hyper parameters : {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt'}
Accuracy : 0.7550757575757576
Test Accuracy: 0.6011904761904762
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 7, 'max_features': 'sqrt'}
Accuracy : 0.7688379267791035
Test Accuracy: 0.31049107142857146
Tuned Hyper parameters : {'criterion': 'log_loss', 'max_depth': 9, 'max_features': 'log2'}
Accuracy : 0.8347998856489423
Test Accuracy: 0.5193032015065913
'''