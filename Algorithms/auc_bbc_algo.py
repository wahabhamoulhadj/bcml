import numpy as np

from Algorithms.boolean_generator_2array import bf_output
from Algorithms.rocch import rocch
from Algorithms.two_d_array_to_fpr_tpr import resp2pts


def auc_bbc(original, soft_detectors, thresholds):
    soft_detectors = np.array(soft_detectors)
    thresholds = np.array(thresholds)
    crisp_detectors = []
    bbc_array = []
    for i, j in zip(soft_detectors, thresholds):
        for k in j:
            crisp_detectors.append(i > k)
            bbc_array.append(i > k)

    for i in crisp_detectors:
        for j in crisp_detectors:
            for k in range(10):
                bbc_array.append(bf_output(k, i, j))

    return rocch(resp2pts(original, bbc_array)[0], resp2pts(original, bbc_array)[1])[2]
