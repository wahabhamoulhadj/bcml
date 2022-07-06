# import pandas as pd
# df = pd.concat([pd.read_csv('../Tables/AUC_Table.csv'),pd.read_csv('../BBC_Algorithms/AUC_Table_BBC.csv') ,pd.read_csv('../IBC_Algorithm/AUC_Table_IBC.csv')], axis=1, join='outer' )
# df.drop(df.filter(regex="Unnamed"),axis=1, inplace=True)
# print(df)
# df.to_csv('../Tables/AUC_Table.csv', index=False)
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from Tables.SixModelsAUC_DF import six_models_prediction_data_frame
from SixModelsAUC_DF import six_models_prediction_data_frame
parameter_grid_models = [{},
                         {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'max_features': ['sqrt', 'log2'],
                          'criterion': ['gini', 'entropy', 'log_loss']},
                         {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'max_features': ['sqrt', 'log2'],
                          'criterion': ['gini', 'entropy', 'log_loss']},
                         {'C': np.logspace(-3, 3, 7), 'solver': ['newton-cg']},
                         {'n_neighbors': list(range(1, 15)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                         {'C': np.logspace(-3, 3, 7), 'kernel': ['linear', 'rbf', 'sigmoid'],
                          'gamma': ['scale', 'auto']}]
models = [GaussianNB(), RandomForestClassifier(), DecisionTreeClassifier(), LogisticRegression(),
          KNeighborsClassifier(), svm.SVC(probability=True)]
dir_path = '../PROMIS/CK_NET/velocity-1.6--CK_NET.csv'
random_variable = 42
print(six_models_prediction_data_frame(models, parameter_grid_models, dir_path, random_variable))