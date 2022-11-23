from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV


def generate_soft_scores(classifier, X_train, X_test, Y_train, Y_test, parameter_grid):
    # Parameter Grid
    clf = GridSearchCV(classifier,  # model
                       param_grid=parameter_grid, refit=True,  # hyper parameters
                       scoring='roc_auc_ovr_weighted',  # metric for scoring
                       cv=5, n_jobs=-1)  # Folds = 5

    clf.fit(X_train, Y_train)  # Training

    # print("Tuned Hyper parameters :", clf.best_params_)
    # print("Predicted probabilities :", 1 - clf.predict_proba(X_test)[:,0])
    # print("Best Score :", clf.best_score_)
    # print("Test Accuracy:", clf.score(X_test, Y_test))
    auc = roc_auc_score(Y_test, clf.best_estimator_.predict_proba(X_test)[:, 1])
    soft_scores = 1 - clf.best_estimator_.predict_proba(X_test)[:, 0]
    # metrics.plot_roc_curve(clf.best_estimator_, X_test, Y_test)
    # plt.show()
    fpr, tpr, thres = roc_curve(Y_test, clf.best_estimator_.predict_proba(X_test)[:, 1])
    # print(len(fpr))
    # print(len(tpr))

    return soft_scores, auc, fpr, tpr
