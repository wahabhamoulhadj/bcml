import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib
from sklearn.model_selection import cross_val_score
matplotlib.rcParams["figure.figsize"] = (20, 10)
path = "CK/*.csv"
for fname in glob.glob(path):

    df1 = pd.read_csv(fname)
    df1['isBug'] = df1['isBug'].map({'YES': 1, 'NO': 0})
    X = df1.drop(['isBug'], axis='columns')
    Y = df1['isBug']
    X = X.to_numpy()
    Y = Y.to_numpy()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 3/10, random_state = 43 )
    folds = StratifiedKFold(n_splits= 5)

    print(cross_val_score(LogisticRegression(), X_train, Y_train, cv=5))
