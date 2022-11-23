"""
WPBC2 Algotihtm

Inputs : Ground Truth, Soft Scores, Threshold Sampling (nb_thresh)
Outputs : AUC of the ROCCH, FPR, TPR of the points
"""

import numpy as np
from matplotlib import pyplot as plt

from BBC_Algorithms.boolean_generator_2array import bf_output
from BBC_Algorithms.rocch import rocch
from BBC_Algorithms.two_d_array_to_fpr_tpr import resp2pts
from IBC_Algorithm.sampleScores import sample_scores
from PBC_Algorithms.MinMaxKappa_Algo import MinMaxKappa
from WPBC2_Algorithms.AUC_ROC import roc_auc
from WPBC2_Algorithms.LinearWeightedKappa import linear_weighted_kappa


def phase_1(original, soft_detectors, k_th, k_ratio, nb_thresh):
    thresh = []  # Threshold array of the entire soft detector array
    for score in soft_detectors:
        thresh.append(sample_scores(score, nb_thresh))  # Sample Score gives us the threshold sample row
        # of the corresponding soft detector score

    auc_all = []
    for i in range(len(soft_detectors)):
        auc_all.append(roc_auc(original, soft_detectors[i], thresh[i])[2])
        # plt.scatter(roc_auc(original, soft_detectors[i], thresh[i])[0], roc_auc(original, soft_detectors[i], thresh[i])[1])
        # plt.show()

    soft_detectors = np.array(soft_detectors)
    original = np.array(original)

    base_array = []

    auc_all = np.array(auc_all)
    # return auc_all

    while len(soft_detectors):

        base_soft_detector = soft_detectors[np.argsort(auc_all)[-1]]
        base_array.append(base_soft_detector)
        soft_detectors = np.delete(soft_detectors, [np.argsort(auc_all)[-1]])
        auc_all = np.delete(auc_all, [np.argsort(auc_all)[-1]])

        delete_indx = []

        for i in range(len(soft_detectors)):
            kp = linear_weighted_kappa(base_soft_detector, soft_detectors[i], nb_thresh)
            if k_th < kp <= 1:
                delete_indx.append(i)
        soft_detectors = np.delete(soft_detectors, [delete_indx])
        auc_all = np.delete(auc_all, [delete_indx])

    thresh = []  # Threshold array of the entire soft detector array
    for score in base_array:
        thresh.append(sample_scores(score, nb_thresh))  # Sample Score gives us the threshold sample row
        # of the corresponding soft detector score

    pruned_crisp_detectors = []
    for i, j in zip(base_array, thresh):
        crisp_detectors = []
        for k in j:
            crisp_detectors.append(i > k)
        for i in MinMaxKappa(original, crisp_detectors, k_ratio):
            pruned_crisp_detectors.append(i)

    bbc_array = []
    for i in pruned_crisp_detectors:
        for j in pruned_crisp_detectors:
            for k in range(10):
                bbc_array.append(bf_output(k, i, j))  # Appends the boolean combination of scores i and j
                # with corresponding boolean combination K

    # resp2pts converts the responses to FPR and TPR
    # rocch inputs the FPR and TPR to give out FPR, TPR of emerging points, AUC and Indexes of the input scores

    roc = rocch(resp2pts(original, bbc_array)[0], resp2pts(original, bbc_array)[1])

    return roc[2], roc[0], roc[1]

#
#
# def WPBC2_Algo(original, soft_detectors, k_ratio, nb_thresh):
#     thresh = []  # Threshold array of the entire soft detector array
#     for score in soft_detectors:
#         thresh.append(sample_scores(score, nb_thresh))  # Sample Score gives us the threshold sample row
#         # of the corresponding soft detector score
#
#     soft_detectors = np.array(soft_detectors)
#     thresholds = np.array(thresh)
#
#     crisp_detectors = []
#
#     bbc_array = []  # It contains all the combinations of Crisp Detectors
#
#     for i, j in zip(soft_detectors, thresholds):
#         for k in j:
#             crisp_detectors.append(i >= k)
#             bbc_array.append(i >= k)
#
#     for i in crisp_detectors:
#         for j in crisp_detectors:
#             for k in range(10):
#                 bbc_array.append(bf_output(k, i, j))  # Appends the boolean combination of scores i and j
#                 # with corresponding boolean combination K
#
#     # resp2pts converts the responses to FPR and TPR
#     # rocch inputs the FPR and TPR to give out FPR, TPR of emerging points, AUC and Indexes of the input scores
#
#     roc = rocch(resp2pts(original, bbc_array)[0], resp2pts(original, bbc_array)[1])
#
#     return roc[2], roc[0], roc[1]
