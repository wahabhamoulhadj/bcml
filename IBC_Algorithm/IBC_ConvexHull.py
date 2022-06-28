import numpy as np

from BBC_Algorithms.rocch import rocch


def check_convhull(NEW, OLD, tol=None):
    if tol is None:
        tol = 0.001

    FP = np.concatenate((NEW[:, 0], OLD[:, 0]))
    TP = np.concatenate((NEW[:, 1], OLD[:, 1]))

    auc = np.trapz(OLD[:, 1], OLD[:, 0])
    FPR, TPR, AUC, _ = rocch(FP, TP)

    if (AUC == 1) or (AUC > (auc + tol)):
        improved = True
        NEW = np.c_[FP, TP]
    else:
        improved = False
        NEW = OLD
        AUC = auc

    return NEW, improved, AUC
