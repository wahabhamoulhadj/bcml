import numpy as np

from BBC_Algorithms.one_d_array_to_fpr_tpr import tpr_fpr


def resp2pts(original, rs):
    fpr_list = np.empty((0, 0), float)
    tpr_list = np.empty((0, 0), float)
    for i in rs:
        tpr, fpr = tpr_fpr(original, i)
        fpr_list = np.append(fpr_list, fpr)
        tpr_list = np.append(tpr_list, tpr)

    return fpr_list, tpr_list

