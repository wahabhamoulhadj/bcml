import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import glob
from sklearn.naive_bayes import GaussianNB
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


parameter_grid = {}
model = GaussianNB()
main(model, parameter_grid)


'''
Tuned Hyper parameters : {}
Accuracy : 0.7795833333333333
Test Accuracy: 0.3653198653198653
Tuned Hyper parameters : {}
Accuracy : 0.5937064884433305
Test Accuracy: 0.4891304347826087
Tuned Hyper parameters : {}
Accuracy : 0.7537237237237238
Test Accuracy: 0.5379746835443038
Tuned Hyper parameters : {}
Accuracy : 0.793993993993994
Test Accuracy: 0.5
Tuned Hyper parameters : {}
Accuracy : 0.5074074074074074
Test Accuracy: 0.11386138613861385
Tuned Hyper parameters : {}
Accuracy : 0.5448532638210057
Test Accuracy: 0.5041666666666667
Tuned Hyper parameters : {}
Accuracy : 0.6627674981783892
Test Accuracy: 0.5089285714285714
Tuned Hyper parameters : {}
Accuracy : 0.5781892164892165
Test Accuracy: 0.5110619469026549
Tuned Hyper parameters : {}
Accuracy : 0.7223533151381627
Test Accuracy: 0.49038461538461536
Tuned Hyper parameters : {}
Accuracy : 0.6721270718232043
Test Accuracy: 0.4975624375624376
Tuned Hyper parameters : {}
Accuracy : 0.7814414954765831
Test Accuracy: 0.5451927337173239
Tuned Hyper parameters : {}
Accuracy : 0.7535437430786269
Test Accuracy: 0.3848797250859106
Tuned Hyper parameters : {}
Accuracy : 0.7117609890109889
Test Accuracy: 0.5107758620689655
Tuned Hyper parameters : {}
Accuracy : 0.6802840909090909
Test Accuracy: 0.5794768611670019
Tuned Hyper parameters : {}
Accuracy : 0.5989169000933707
Test Accuracy: 0.505712866368604
Tuned Hyper parameters : {}
Accuracy : 0.7194083694083694
Test Accuracy: 0.5
Tuned Hyper parameters : {}
Accuracy : 0.39192273924495175
Test Accuracy: 0.35747126436781607
Tuned Hyper parameters : {}
Accuracy : 0.8988278388278388
Test Accuracy: 0.5
Tuned Hyper parameters : {}
Accuracy : 0.6736996400789504
Test Accuracy: 0.5026455026455027
Tuned Hyper parameters : {}
Accuracy : 0.641668783068783
Test Accuracy: 0.48904865914330786
Tuned Hyper parameters : {}
Accuracy : 0.6666987179487179
Test Accuracy: 0.5
Tuned Hyper parameters : {}
Accuracy : 0.6666666666666667
Test Accuracy: 0.5434782608695652
Tuned Hyper parameters : {}
Accuracy : 0.6211572244180941
Test Accuracy: 0.48464912280701755
Tuned Hyper parameters : {}
Accuracy : 0.7634823232323231
Test Accuracy: 0.48214285714285715
Tuned Hyper parameters : {}
Accuracy : 0.6440584415584416
Test Accuracy: 0.5
Tuned Hyper parameters : {}
Accuracy : 0.5398573975044563
Test Accuracy: 0.45401785714285714
Tuned Hyper parameters : {}
Accuracy : 0.6568921923638905
Test Accuracy: 0.5127118644067796

'''