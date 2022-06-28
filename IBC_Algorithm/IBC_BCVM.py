import numpy as np

from BBC_Algorithms.rocch import rocch
from IBC_Algorithm.IBC_BCVRT import bcvrt
from IBC_Algorithm.IBC_BCVTT import bcvtt
from IBC_Algorithm.IBC_mrroc_n import mrroc_n
from IBC_Algorithm.sampleScores import sample_scores


def bcvm(scores, lab, nb_thresh, fun):
    thresh = 0
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

    ncurves = len(scores)
    fpr = np.array([])
    tpr = np.array([])
    ttb = {}
    ttb[1] = {}

    ttb[1][1] = nb_thresh

    RS, t1, t2, bf, fp, tp, au = bcvtt(scores[0], thresh[0], scores[1], thresh[1], lab, fun)

    fpr = np.concatenate((fpr, fp))
    tpr = np.concatenate((tpr, tp))
    ttb[1][2] = {}
    ttb[1][2]['t1'] = t1  # t1 thresholds on ROC 1.
    ttb[1][2]['t2'] = t2  # t2 thresholds on ROC 2.
    ttb[1][2]['bf'] = bf  # corresponding Boolean functions.

    for i in range(2, len(scores)):
        RS, t1, t2, bf, fp, tp, au = bcvrt(RS, [], scores[i], thresh[i], lab, fun)
        fpr = np.concatenate((fpr, fp))
        tpr = np.concatenate((tpr, tp))
        ttb[1][i + 1] = {}
        ttb[1][i + 1]['t1'] = t1  # t1 thresholds on ROC 1.
        ttb[1][i + 1]['t2'] = t2  # t2 thresholds on ROC 2.
        ttb[1][i + 1]['bf'] = bf  # corresponding Boolean functions.

    rs = RS
    fpr, tpr, auc, _ = rocch(fpr, tpr)

    fpc, tpc, _ = mrroc_n(scores, lab, nb_thresh)
    fpr = np.concatenate((fpr, fp))
    tpr = np.concatenate((tpr, tp))
    fpr, tpr, auc, _ = rocch(fpr, tpr)

    return fpr, tpr, auc, ttb, rs
