import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def split_table(fname, random_variable):
    df1 = pd.read_csv(fname)
    df1.isBug = df1.isBug.map(dict(YES=1, NO=0))

    X = df1.iloc[:, :-1]  # X contains the features
    Y = df1.iloc[:, -1:]  # Y is the target variable

    X = X.to_numpy()
    Y = Y.to_numpy().ravel()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=3 / 10, random_state=random_variable)

    fit_param = MinMaxScaler().fit(X_train) #Pre Processing the TRAINING DATA to avoid data leakage
    X_train = fit_param.transform(X_train)
    X_test = fit_param.transform(X_test)

    return X_train, X_test, Y_train, Y_test