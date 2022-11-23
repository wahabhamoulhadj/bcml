import pandas as pd
import numpy as np

from BBC_Algorithms.boolean_generator_2array import bf_output
from BBC_Algorithms.one_d_array_to_fpr_tpr import tpr_fpr
from BBC_Algorithms.rocch import rocch
from BBC_Algorithms.two_d_array_to_fpr_tpr import resp2pts
from IBC_Algorithm.sampleScores import sample_scores

import glob

from Tables.splitTable import split_table


def combine_lists(list_1, list_2):
    ans = []
    for i in range(len(list_1)):
        ans.append([list_1[i], list_2[i]])
    return ans


def roc_old(score_1, thresh_1, score_2, thresh_2, labels):
    original_crisp_detectors = []
    for i in thresh_1:
        original_crisp_detectors.append(score_1 >= i)
    for i in thresh_2:
        original_crisp_detectors.append(score_2 >= i)

    return rocch(resp2pts(labels, original_crisp_detectors)[0], resp2pts(labels, original_crisp_detectors)[1])


def roc_old_bcm_all(R, score_2, thresh_2, labels):
    original_crisp_detectors = []
    for i in R:
        original_crisp_detectors.append(i)
    for i in thresh_2:
        original_crisp_detectors.append(score_2 >= i)

    return rocch(resp2pts(labels, original_crisp_detectors)[0], resp2pts(labels, original_crisp_detectors)[1])


def list_difference(list_1, list_2, list_idx):
    list_difference_arr = []
    list_difference_idx = []
    for count, element in enumerate(list_1):
        if element not in list_2:
            list_difference_arr.append(element)
            list_difference_idx.append(list_idx[count])

    return list_difference_arr, list_difference_idx


def bc_all(score_1, thresh_1, score_2, thresh_2, labels):
    F = []
    R = []
    S_Global = []

    rocch_old = roc_old(score_1, thresh_1, score_2, thresh_2, labels)

    for k in range(10):
        for i in thresh_1:
            r_a = score_1 >= i
            for j in thresh_2:
                r_b = score_2 >= j
                r_c = bf_output(k, r_a, r_b)
                tpr, fpr = tpr_fpr(labels, r_c)
                F = list(F)
                F.append([fpr, tpr])

        F = np.array(F)
        # print(len(F))
        rocch_new = rocch(F[:, 0], F[:, 1])

        emerging_fpr_tpr, emerging_idx = list_difference(combine_lists(rocch_new[0], rocch_new[1]),
                                                         combine_lists(rocch_old[0], rocch_old[1]),
                                                         rocch_new[3])
        # print(F[emerging_idx])
        for i in range(len(emerging_idx)):
            S_Global.append(
                [[score_1, thresh_1[int(((emerging_idx[i] % (len(thresh_1) * len(thresh_2))) / len(thresh_2)))]],
                 [score_2, thresh_2[int((emerging_idx[i] % (len(thresh_1) * len(thresh_2))) % len(thresh_2))]],
                 k])
            R.append(bf_output(k, score_1 >= thresh_1[
                int((emerging_idx[i] % (len(thresh_1) * len(thresh_2))) / len(thresh_2))],
                               score_2 >= thresh_2[
                                   int((emerging_idx[i] % (len(thresh_1) * len(thresh_2))) % len(thresh_2))]))

        rocch_old = rocch_new

    return rocch_old, R, S_Global


def bcm_all(rocch_old, R, S_Global_old, score_2, thresh_2, labels):
    F = []
    S_Global_new = []

    for k in range(10):
        for i in R:
            r_a = i
            for j in thresh_2:
                r_b = score_2 >= j
                r_c = bf_output(k, r_a, r_b)
                tpr, fpr = tpr_fpr(labels, r_c)
                F = list(F)
                F.append([fpr, tpr])

        F = np.array(F)
        F = np.array(F)

        if len(F) != 0:
            rocch_new = rocch(F[:, 0], F[:, 1])
            emerging_fpr_tpr, emerging_idx = list_difference([rocch_new[0], rocch_new[1]], [rocch_old[0], rocch_old[1]],
                                                             rocch_new[3])
            # print(F[emerging_idx])
            for i in range(len(emerging_idx)):
                S_Global_new.append(
                    [S_Global_old[int((emerging_idx[i] % (len(S_Global_old) * len(thresh_2))) / len(thresh_2))],
                     [score_2, thresh_2[int((emerging_idx[i] % (len(S_Global_old) * len(thresh_2))) % len(thresh_2))]],
                     k])
                R.append(bf_output(k, R[int((emerging_idx[i] % (len(S_Global_old) * len(thresh_2))) / len(thresh_2))],
                                   score_2 >= thresh_2[
                                       int((emerging_idx[i] % (len(S_Global_old) * len(thresh_2))) % len(thresh_2))]))

            rocch_old = rocch_new

    return rocch_old, R, S_Global_new


def IBC_Pseudo_Algo(original, soft_detectors, nb_thresh=None, thresh=None, max_iter=20, tol=0.001):
    if nb_thresh is None:
        pass
    if thresh is None:
        thresh = []  # Threshold array of the entire soft detector array
        for score in soft_detectors:
            thresh.append(sample_scores(score, nb_thresh))  # Sample Score gives us the threshold sample row
            # of the corresponding soft detector score

    soft_detectors = np.array(soft_detectors)
    thresholds = np.array(thresh)

    x, y, z = bc_all(soft_detectors[0], thresh[0], soft_detectors[1], thresh[1], original)
    for i in range(2, len(soft_detectors)):
        x, y, z = bcm_all(x, y, z, soft_detectors[i], thresh[i], original)

    auc_inc = 1
    i = 0

    while i <= max_iter and auc_inc <= tol:
        auc_old = x[2]
        for j in range(len(soft_detectors)):
            x, y, z = bcm_all(x, y, z, soft_detectors[j], thresh[j], original)
        auc_new = x[2]
        auc_inc = auc_new - auc_old
        i += 1

    return x, y, z

def testing():
    path = "../PROMIS/CK_NET_PROC/*.csv"
    ibc_list = []
    ibc_fpr = []
    ibc_tpr = []
    ibc_idx = []
    all_predict_proba = np.load(
        '../Results_Dataframes/CK_NET_PROC_Results/CK_NET_PROC_all_six_models_predict_proba.npy', allow_pickle=True)
    for count, fname in enumerate(glob.glob(path)):
        soft_detectors = all_predict_proba[count]
        IBCvr = IBC_Pseudo_Algo(split_table(fname, 42)[3], soft_detectors, 12)
        ibc_list.append(IBCvr[0])
        ibc_fpr.append(IBCvr[1])
        ibc_tpr.append(IBCvr[2])
        # ibc_idx.append(IBCvr[3])
        # print(count, ibc_list)
        # print(count, ibc_fpr)
        # print(count, ibc_tpr)
        print('-----------------------')
        print(count)
        print(IBCvr[0][2])
        # print(IBCvr[1])
        # print(IBCvr[2])
        print('-----------------------')
        # if count == 0:
        #     break

    # columns = ['IBC']
    # # np.save('CK_NET_PROC_ibc_roc_fpr', np.array(ibc_fpr, dtype=object))
    # # np.save('CK_NET_PROC_ibc_roc_tpr', np.array(ibc_tpr, dtype=object))
    # df = pd.DataFrame(ibc_list, columns=columns)
    # print(df)

    # df.to_csv('CK_NET_PROC_AUC_Table_IBC.csv')


# testing()
