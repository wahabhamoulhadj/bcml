import numpy as np


def roc(fpr0, tpr0):

    vert = np.vstack((fpr0, tpr0)).T[np.lexsort((tpr0, fpr0))]
    F = []
    T = []
    for v in vert:
        F += [v[0]]
        T += [v[1]]
    auc = np.trapz(T, F)

    return F, T, auc
