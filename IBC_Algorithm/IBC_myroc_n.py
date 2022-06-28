import numpy as np

from BBC_Algorithms.one_d_array_to_fpr_tpr import tpr_fpr
from BBC_Algorithms.rocch import rocch
from IBC_Algorithm.sampleScores import sample_scores


def myroc_n(scores, lab, nb_thresh=None):
    thresh = sample_scores(scores, nb_thresh)
    fpc = np.array([])
    tpc = np.array([])
    for i in thresh:
        tp, fp = tpr_fpr(lab, scores > i)
        tpc = np.append(tpc, tp)
        fpc = np.append(fpc, fp)

    auch = np.trapz(tpc, fpc)
    return fpc, tpc, auch, thresh
