"""
WPBC2 Algotihtm

Inputs : Ground Truth, Soft Scores, Threshold Sampling (nb_thresh)
Outputs : AUC of the ROCCH, FPR, TPR of the points
"""
import glob

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score
from IBC_Algorithm.IBC_Pseudo_ALgo import IBC_Pseudo_Algo
from IBC_Algorithm.sampleScores import sample_scores

from Tables.splitTable import split_table
from WPBC2_Algorithms.AUC_ROC import roc_auc
from WPBC2_Algorithms.LinearWeightedKappa import linear_weighted_kappa


def phase_3(original, soft_detectors, k_th = 0.5, k_ratio = 0.5 , nb_thresh = 12):
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

    new_thresh = []

    for i, j in zip(base_array, thresh):
        crisp_detectors = []
        mini_thres = []
        for k in j:
            crisp_detectors.append(i > k)
            mini_thres.append(k)

        k_ratio_array = []
        pbc_array = []
        for p in crisp_detectors:
            k_ratio_array.append(cohen_kappa_score(original, p))

        for p in range(int((k_ratio / 2) * len(np.argsort(k_ratio_array)))):
            mini_thres.append(mini_thres[np.argsort(k_ratio_array)[p]])

        for p in range(int(len(crisp_detectors) - ((k_ratio / 2) * len(crisp_detectors))), len(crisp_detectors)):
            mini_thres.append(mini_thres[np.argsort(k_ratio_array)[p]])

        new_thresh.append(mini_thres)


    return IBC_Pseudo_Algo(original, base_array ,None ,new_thresh )


def testing():
    path = "../PROMIS/CK_NET_PROC/*.csv"
    ibc_list = []
    wpibc_fpr = []
    wpibc_tpr = []

    all_predict_proba = np.load(
        '../Results_Dataframes/CK_NET_PROC_Results/CK_NET_PROC_all_six_models_predict_proba.npy', allow_pickle=True)
    for count, fname in enumerate(glob.glob(path)):
        soft_detectors = all_predict_proba[count]
        WPIBC_algo = phase_3(split_table(fname, 42)[3], soft_detectors)
        ibc_list.append(WPIBC_algo[0])
        wpibc_fpr.append(WPIBC_algo[1])
        wpibc_tpr.append(WPIBC_algo[2])

        print('-----------------------')
        print(count)
        print(WPIBC_algo[0][2])

        print('-----------------------')

# testing()

