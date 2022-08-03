"""
PBC Algotihtm

Inputs : Ground Truth, Soft Scores, K_ratio, Threshold Sampling (nb_thresh)
Outputs : AUC of the ROCCH, FPR, TPR of the points
"""

import numpy as np

from BBC_Algorithms.boolean_generator_2array import bf_output
from BBC_Algorithms.rocch import rocch
from BBC_Algorithms.two_d_array_to_fpr_tpr import resp2pts
from IBC_Algorithm.sampleScores import sample_scores
from sklearn.metrics import cohen_kappa_score

from PBC_Algorithms.MinMaxKappa_Algo import MinMaxKappa


def pbc_Algo(original, soft_detectors, k_ratio, nb_thresh):
    thresh = []  # Threshold array of the entire soft detector array
    for score in soft_detectors:
        thresh.append(sample_scores(score, nb_thresh))  # Sample Score gives us the threshold sample row
        # of the corresponding soft detector score

    soft_detectors = np.array(soft_detectors)
    thresholds = np.array(thresh)

    crisp_detectors = []
    k_ratio_array = []
    pbc_array = []  # stores the pruned crisp detectors

    for i, j in zip(soft_detectors, thresholds):
        for k in j:
            crisp_detectors.append(i > k)

    pbc_array = MinMaxKappa(original, crisp_detectors, k_ratio)

    bbc_array = []

    for i in pbc_array:
        for j in pbc_array:
            for k in range(10):
                bbc_array.append(bf_output(k, i, j))  # Appends the boolean combination of scores i and j
                # with corresponding boolean combination K

    # resp2pts converts the responses to FPR and TPR
    # rocch inputs the FPR and TPR to give out FPR, TPR of emerging points, AUC and Indexes of the input scores

    roc = rocch(resp2pts(original, bbc_array)[0], resp2pts(original, bbc_array)[1])

    return roc[2], roc[0], roc[1]
