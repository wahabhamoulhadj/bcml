import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib
from sklearn.model_selection import cross_val_score

matplotlib.rcParams["figure.figsize"] = (20, 10)
path = "CK/*.csv"
# for fname in glob.glob(path):
#
#     df1 = pd.read_csv(fname)
#     df1['isBug'] = df1['isBug'].map({'YES': 1, 'NO': 0})
#     X = df1.drop(['isBug'], axis='columns')
#     Y = df1['isBug']
#     X = X.to_numpy()
#     Y = Y.to_numpy()
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 3/10, random_state = 43 )
# cv = StratifiedKFold(n_splits=5)
# classifier = LogisticRegression()
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)
# fig, ax = plt.subplots()
#
# for i, (train, test) in enumerate(cv.split(X_train, Y_train)):
#     classifier.fit(X_train[train], Y_train[train])
#     viz = RocCurveDisplay.from_estimator(
#         classifier,
#         X_train[train],
#         Y_train[train],
#         name="ROC fold {}".format(i), alpha= 3/10, lw=1,
#         ax=ax,
#     )
#     interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#     interp_tpr[0] = 0.0
#     tprs.append(interp_tpr)
#     aucs.append(viz.roc_auc)
# ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
#
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# ax.plot(
#     mean_fpr,
#     mean_tpr,
#     color="b",
#     label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
#     lw=2,
#     alpha=0.8,
# )
#
# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(
#     mean_fpr,
#     tprs_lower,
#     tprs_upper,
#     color="grey",
#     alpha=0.2,
#     label=r"$\pm$ 1 std. dev.",
# )
#
# ax.set(
#     xlim=[-0.05, 1.05],
#     ylim=[-0.05, 1.05],
#     title="Receiver operating characteristic example",
# )
# ax.legend(loc="lower right")
# plt.show()
# print(cross_val_score(classifier, X_train, Y_train, cv=5))

for fname in glob.glob(path):

    df1 = pd.read_csv(fname)

    df1['isBug'] = df1['isBug'].map({'YES': 1, 'NO': 0})  # Encoding isBig Coloumn

    df1.iloc[:,-1:].applymap(lambda x: {'YES': 1, 'NO': 0})
    X = df1.iloc[:,:-1] # X cointains the features
    Y = df1.iloc[:,-1:]  # Y is the target variable

    X = X.to_numpy()
    Y = Y.to_numpy().ravel()

    # Train Test Split Data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=3 / 10, random_state=43)

    # Pre Processing just X_train to avoid Data Leakage
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    # Parameter Grid
    parameters = {'C': np.logspace(-3, 3, 7),
                  'solver': ['newton-cg', 'lbfgs']}

    classifier = LogisticRegression()
    clf = GridSearchCV(classifier,  # model
                       param_grid=parameters, refit=True,  # hyperparameters
                       scoring='roc_auc_ovr_weighted',  # metric for scoring
                       cv=5, n_jobs=-1)  # Folds = 5

    clf.fit(X_train, Y_train) # Training

    print("Tuned Hyper parameters :", clf.best_params_)
    print("Accuracy :", clf.best_score_)

    print("Test Accuracy:", clf.score(X_test, Y_test))
