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
                         {'C': np.logspace(-3, 3, 7), 'solver': ['newton-cg']},
                         {'n_neighbors': list(range(1, 15)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                         {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001],
                          'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}]
models = [GaussianNB(), RandomForestClassifier(), DecisionTreeClassifier(), LogisticRegression(),
          KNeighborsClassifier(), svm.SVC(probability=True)]

dir_path = "../PROMIS/CK_NET/*.csv"  # Path to dataset

random_variable = 42  # For a fixed random seed
smpdf = six_models_prediction_data_frame(models, parameter_grid_models, dir_path, random_variable)
# Saving the predict probability distributions in an array to decrease O(n).

np.save('../Results_Dataframes/CK_NET_Results/CK_NET_all_six_models_predict_proba', np.array(smpdf[0], dtype=object))
np.save('../Results_Dataframes/CK_NET_Results/CK_NET_all_six_models_fpr', np.array(smpdf[2], dtype=object))
np.save('../Results_Dataframes/CK_NET_Results/CK_NET_all_six_models_tpr', np.array(smpdf[3], dtype=object))

# Creating the Data Frame and Saving to 'AUC_Table')
columns = ['NaiveBayes', 'RandomForest', 'DecisionTree', 'LogisticRegression', 'KNN', 'SVM']
df = pd.DataFrame(smpdf[1], columns=columns)

df.to_csv('../Results_Dataframes/CK_NET_Results/CK_NET_all_six_models_auc.csv')

# smpdf = six_models_prediction_data_frame(models, parameter_grid_models, dir_path, random_variable)
# # Saving the predict probability distributions in an array to decrease O(n).
#
# np.save('all_six_models_predict_proba', np.array(smpdf[0], dtype=object))
# np.save('all_six_models_fpr', np.array(smpdf[2], dtype=object))
# np.save('all_six_models_tpr', np.array(smpdf[3], dtype=object))
#
# # Creating the Data Frame and Saving to 'AUC_Table')
# columns = ['NaiveBayes', 'RandomForest', 'DecisionTree', 'LogisticRegression', 'KNN', 'SVM']
# df = pd.DataFrame(smpdf[1],columns=columns)
#
# df.to_csv('AUC_Table.csv')
