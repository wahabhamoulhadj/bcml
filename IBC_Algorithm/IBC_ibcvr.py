import numpy as np

from BBC_Algorithms.rocch import rocch
from BBC_Algorithms.two_d_array_to_fpr_tpr import resp2pts
from IBC_Algorithm.IBC_BCVM import bcvm
from IBC_Algorithm.IBC_BCVRT import bcvrt
from IBC_Algorithm.IBC_ConvexHull import check_convhull
from IBC_Algorithm.IBC_mrroc_n import mrroc_n
from IBC_Algorithm.sampleScores import sample_scores
from Tables.splitTable import split_table


def ibcvr(scores, lab, nb_thresh=None, max_iter=None, tol=None, fun=None):
    if nb_thresh is None:
        nb_thresh = []
        thresh = scores
    else:
        try:
            if len(nb_thresh) == 0:
                nb_thresh = []
                thresh = scores
        except:
            thresh = []
            for score in scores:
                thresh.append(sample_scores(score, nb_thresh))

    if max_iter is None:
        max_iter = 20
    if tol is None:
        tol = 0.001
    if fun is None:
        fun = range(10)

    fp, tp, auc, ttb, r_prev = bcvm(scores, lab, nb_thresh, fun)

    # ch_prev=[fp,tp];
    ch_prev = np.c_[fp, tp]

    for i in range(1, max_iter):
        r_new = np.empty((0, len(lab)), int)
        ttb[i + 1] = {}
        for n in range(len(scores)):
            s = scores[n]
            t = thresh[n]
            rs, IR, TH, BF, _, _, _ = bcvrt(r_prev, [], s, t, lab, fun)

            r_new = np.concatenate((r_new, rs))
            ttb[i + 1][n + 1] = {}
            ttb[i + 1][n + 1]['ir'] = IR  # t1 thresholds on ROC 1.
            ttb[i + 1][n + 1]['th'] = TH  # t2 thresholds on ROC 2.
            ttb[i + 1][n + 1]['bf'] = BF  # corresponding Boolean functions.

        ff, tt = resp2pts(lab, r_new)
        FP, TP, AA, IX = rocch(ff, tt)
        ch_new = np.c_[FP, TP]

        ch_new, improved, auc = check_convhull(ch_new, ch_prev, tol)

        if improved is True:
            ch_prev = ch_new
            r_prev = r_new[IX]
        else:
            ttb[i + 1] = {}
            break
    rs = r_prev
    fpr = ch_prev[:, 0]
    tpr = ch_prev[:, 1]

    fpc, tpc, _ = mrroc_n(scores, lab, nb_thresh)
    tpr = np.concatenate((tpr, tpc))
    fpr = np.concatenate((fpr, fpc))

    fpr, tpr, auc, _ = rocch(fpr, tpr)

    return fpr, tpr, auc, ttb, rs


# print(ibcvr(np.load('../Tables/CK_all_six_models_predict_proba.npy', allow_pickle=True)[0],
#             split_table('../PROMIS/CK/ant-1.3--CK_all_auc.csv', 42,)[3], 12))
