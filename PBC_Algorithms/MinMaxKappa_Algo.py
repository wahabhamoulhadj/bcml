import numpy as np
from sklearn.metrics import cohen_kappa_score


def MinMaxKappa(original, crisp_detectors, k_ratio):
    k_ratio_array = []
    pbc_array = []
    for i in crisp_detectors:
        k_ratio_array.append(cohen_kappa_score(original, i))

    crisp_detectors = np.array(crisp_detectors)
    k_ratio_array = np.array(k_ratio_array)
    crisp_detectors = crisp_detectors[np.argsort(k_ratio_array)]

    for i in range(int((k_ratio / 2) * len(crisp_detectors))):
        pbc_array.append(crisp_detectors[i])

    for i in range(int(len(crisp_detectors) - ((k_ratio / 2) * len(crisp_detectors))), len(crisp_detectors)):
        pbc_array.append(crisp_detectors[i])

    return pbc_array
