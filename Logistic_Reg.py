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


parameter_grid = {'C': np.logspace(-3, 3, 7), 'solver': ['newton-cg', 'lbfgs']}
model = LogisticRegression()
main(model, parameter_grid)

'''
Tuned Hyper parameters : {'C': 1.0, 'solver': 'newton-cg'}
Accuracy : 0.8658333333333333
Test Accuracy: 0.537037037037037
Tuned Hyper parameters : {'C': 1.0, 'solver': 'newton-cg'}
Accuracy : 0.5909913673071567
Test Accuracy: 0.5407608695652174
Tuned Hyper parameters : {'C': 100.0, 'solver': 'newton-cg'}
Accuracy : 0.8632732732732732
Test Accuracy: 0.22151898734177217
Tuned Hyper parameters : {'C': 10.0, 'solver': 'newton-cg'}
Accuracy : 0.8154423654423655
Test Accuracy: 0.5733333333333333
Tuned Hyper parameters : {'C': 0.1, 'solver': 'newton-cg'}
Accuracy : 0.7488888888888889
Test Accuracy: 0.693069306930693
Tuned Hyper parameters : {'C': 0.01, 'solver': 'newton-cg'}
Accuracy : 0.5664676876289779
Test Accuracy: 0.5848544973544973
Tuned Hyper parameters : {'C': 100.0, 'solver': 'newton-cg'}
Accuracy : 0.7121701384424158
Test Accuracy: 0.45782424812030076
Tuned Hyper parameters : {'C': 0.1, 'solver': 'newton-cg'}
Accuracy : 0.6306844935844935
Test Accuracy: 0.6159430309734514
Tuned Hyper parameters : {'C': 10.0, 'solver': 'newton-cg'}
Accuracy : 0.7654001293391646
Test Accuracy: 0.5006978908188585
Tuned Hyper parameters : {'C': 10.0, 'solver': 'newton-cg'}
Accuracy : 0.7164314593558259
Test Accuracy: 0.5456543456543457
Tuned Hyper parameters : {'C': 10.0, 'solver': 'newton-cg'}
Accuracy : 0.8340547475050399
Test Accuracy: 0.6085511741249447
Tuned Hyper parameters : {'C': 0.001, 'solver': 'newton-cg'}
Accuracy : 0.83421926910299
Test Accuracy: 0.7857961053837343
Tuned Hyper parameters : {'C': 100.0, 'solver': 'newton-cg'}
Accuracy : 0.8205384615384617
Test Accuracy: 0.37284482758620685
Tuned Hyper parameters : {'C': 100.0, 'solver': 'newton-cg'}
Accuracy : 0.7615909090909092
Test Accuracy: 0.33970489604292425
Tuned Hyper parameters : {'C': 100.0, 'solver': 'newton-cg'}
Accuracy : 0.7884126984126985
Test Accuracy: 0.33606557377049184
Tuned Hyper parameters : {'C': 1000.0, 'solver': 'newton-cg'}
Accuracy : 0.8288023088023087
Test Accuracy: 0.2582417582417582
Tuned Hyper parameters : {'C': 100.0, 'solver': 'newton-cg'}
Accuracy : 0.416198419666374
Test Accuracy: 0.2712643678160919
Tuned Hyper parameters : {'C': 10.0, 'solver': 'newton-cg'}
Accuracy : 0.8768498168498169
Test Accuracy: 0.5441176470588236
Tuned Hyper parameters : {'C': 100.0, 'solver': 'newton-cg'}
Accuracy : 0.7295193312434692
Test Accuracy: 0.5211640211640212
Tuned Hyper parameters : {'C': 1000.0, 'solver': 'newton-cg'}
Accuracy : 0.7179305114638448
Test Accuracy: 0.45687719936900867
Tuned Hyper parameters : {'C': 1.0, 'solver': 'newton-cg'}
Accuracy : 0.6937499999999999
Test Accuracy: 0.575
Tuned Hyper parameters : {'C': 0.001, 'solver': 'newton-cg'}
Accuracy : 0.7964912280701755
Test Accuracy: 0.9782608695652174
Tuned Hyper parameters : {'C': 0.001, 'solver': 'newton-cg'}
Accuracy : 0.6967171717171717
Test Accuracy: 0.6754385964912281
Tuned Hyper parameters : {'C': 10.0, 'solver': 'newton-cg'}
Accuracy : 0.7902828282828283
Test Accuracy: 0.5510204081632653
Tuned Hyper parameters : {'C': 0.001, 'solver': 'newton-cg'}
Accuracy : 0.7239393939393939
Test Accuracy: 0.6979166666666666
Tuned Hyper parameters : {'C': 1000.0, 'solver': 'newton-cg'}
Accuracy : 0.5553503359385712
Test Accuracy: 0.4747767857142857
Tuned Hyper parameters : {'C': 100.0, 'solver': 'newton-cg'}
Accuracy : 0.7394034686487515
Test Accuracy: 0.5699152542372882
'''
