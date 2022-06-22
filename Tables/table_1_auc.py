import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import csv
import pandas as pd
import glob



def generate_soft_scores(classifier, X_train, X_test, Y_train, Y_test, parameter_grid):
    # Parameter Grid
    clf = GridSearchCV(classifier,  # model
                       param_grid=parameter_grid, refit=True,  # hyperparameters
                       scoring='roc_auc_ovr_weighted',  # metric for scoring
                       cv=5, n_jobs=-1)  # Folds = 5

    clf.fit(X_train, Y_train)  # Training

    # print("Tuned Hyper parameters :", clf.best_params_)
    # print("Predicted probabilities :", 1 - clf.predict_proba(fit_param.transform(X_test))[:,0])
    # print("Best Score :", clf.best_score_)
    # print("Test Accuracy:", clf.score(fit_param.transform(X_test), Y_test))
    auc = roc_auc_score(Y_test, clf.predict_proba(X_test)[:, 1])
    soft_scores = 1 - clf.predict_proba(X_test)[:, 0]
    return soft_scores, auc


def split_table(fname, random_variable):
    df1 = pd.read_csv(fname)
    df1.isBug = df1.isBug.map(dict(YES=1, NO=0))

    X = df1.iloc[:, :-1]  # X contains the features
    Y = df1.iloc[:, -1:]  # Y is the target variable

    X = X.to_numpy()
    Y = Y.to_numpy().ravel()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=3 / 10, random_state=random_variable)

    fit_param = MinMaxScaler().fit(X_train)
    X_train = fit_param.transform(X_train)
    X_test = fit_param.transform(X_test)

    return X_train, X_test, Y_train, Y_test


parameter_grid_models = [{},
                         {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'max_features': ['sqrt', 'log2'],
                          'criterion': ['gini', 'entropy', 'log_loss']},
                         {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'max_features': ['sqrt', 'log2'],
                          'criterion': ['gini', 'entropy', 'log_loss']},
                         {'C': np.logspace(-3, 3, 7), 'solver': ['newton-cg', 'lbfgs']},
                         {'n_neighbors': list(range(1, 15)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                         {'C': np.logspace(-3, 3, 7), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                          'gamma': ['scale', 'auto']}]
models = [GaussianNB(), RandomForestClassifier(), DecisionTreeClassifier(), LogisticRegression(),
          KNeighborsClassifier(), svm.SVC(probability=True)]

path = "../PROMIS/CK/*.csv"

random_variable = 42
dataframe_list = []

all_six_models_predict_proba = []


for count, fname in enumerate(glob.glob(path)):
    rows = []
    predict_proba_rows = []
    X_train, X_test, Y_train, Y_test = split_table(fname, random_variable)
    for model, parameter_grid in zip(models, parameter_grid_models):
        auc = generate_soft_scores(model, X_train, X_test, Y_train, Y_test,
                                   parameter_grid)[1]
        rows.append(auc)
        predict_proba_rows.append(generate_soft_scores(model, X_train, X_test, Y_train, Y_test,
                                                       parameter_grid)[0])
        print(auc)

    dataframe_list.append(rows)
    all_six_models_predict_proba.append(predict_proba_rows)

np.save('all_six_models_predict_proba', np.array(all_six_models_predict_proba, dtype=object))
columns = ['NaiveBayes', 'RandomForest', 'DecisionTree', 'LogisticRegression', 'KNN', 'SVM']
df = pd.DataFrame(dataframe_list, columns=columns)

df.to_csv('AUC_Table.csv')
