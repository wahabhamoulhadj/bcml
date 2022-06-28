import numpy as np
from scipy.spatial import ConvexHull


def rocch(fpr0, tpr0):
    idx = []
    npts = len(fpr0)
    fpr = np.array(list(fpr0) + [1])
    tpr = np.array(list(tpr0) + [0])
    hull = ConvexHull(np.vstack((fpr, tpr)).T)
    vert = hull.vertices
    vert = vert[np.lexsort((tpr[vert], fpr[vert]))]

    # F = [0]
    # T = [0]
    F = []
    T = []
    for v in vert:
        ft = (fpr[v], tpr[v])
        if ft == (1, 0):
            continue
        F += [fpr[v]]
        T += [tpr[v]]
        idx.append(v)
    # F+=[1]
    # T+=[1]
    auc = np.trapz(T, F)

    return F, T, auc, idx
