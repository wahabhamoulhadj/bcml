import numpy as np

from BBC_Algorithms.roc import roc
from BBC_Algorithms.two_d_array_to_fpr_tpr import resp2pts


def roc_auc(original, soft_detector_score, thresholds):

    crisp_detectors = [] # It contains all the combinations of Crisp Detectors
    thresholds = np.array(thresholds)
    soft_detector_score = np.array(soft_detector_score)
    for i in thresholds:
        crisp_detectors.append(soft_detector_score >= i)

    fpr_list, tpr_list = resp2pts(original, crisp_detectors)

    return roc(fpr_list, tpr_list)[0], roc(fpr_list, tpr_list)[1], roc(fpr_list, tpr_list)[2]


