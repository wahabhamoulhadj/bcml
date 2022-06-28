"""#
Program for obtaining the AUC scores of 6 models i.e, 'NaiveBayes'
'RandomForest', 'DecisionTree', 'LogisticRegression', 'KNN', 'SVM'
 on 27 different DataSet

 1. Set the Parameter Grid of Every Model in a Dict.
 2. Set the models in a Dict
 3.
 4.
 5.
 6.
 7.
 8.
 9.
10.

#
#"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from Tables.SixModelsAUC_DF import six_models_prediction_data_frame

# Parameter Grid For Cross Validation
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

dir_path = "../PROMIS/CK/*.csv"  # Path to dataset

random_variable = 42  # For a fixed random seed

# Saving the predict probability distributions in an array to decrease O(n).

np.save('all_six_models_predict_proba', np.array(six_models_prediction_data_frame(models,
                                                                                  parameter_grid_models, dir_path,
                                                                                  random_variable)[0],
                                                 dtype=object))

# Creating the Data Frame and Saving to 'AUC_Table')
columns = ['NaiveBayes', 'RandomForest', 'DecisionTree', 'LogisticRegression', 'KNN', 'SVM']
df = pd.DataFrame(six_models_prediction_data_frame(models, parameter_grid_models, dir_path, random_variable)[1],
                  columns=columns)

df.to_csv('AUC_Table.csv')
