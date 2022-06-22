import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull

from Algorithms.auc_bbc_algo import auc_bbc


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


path = "../PROMIS/CK/*.csv"
thres = [[-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1, 1.1] for i in range(6)]
bbc_list = []
for count, fname in enumerate(glob.glob(path)):
    soft_detectors = np.load('../Tables/all_six_models_predict_proba.npy', allow_pickle=True)[count]
    bbc_list.append(auc_bbc(split_table(fname, 42)[3], soft_detectors,thres))
    print(count, bbc_list )

columns = ['BBC']
df = pd.DataFrame(bbc_list, columns=columns)

df.to_csv('AUC_Table_BBC.csv')
