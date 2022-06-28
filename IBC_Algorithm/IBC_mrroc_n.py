import numpy as np

from BBC_Algorithms.rocch import rocch
from IBC_Algorithm.IBC_myroc_n import myroc_n


def mrroc_n(scores, lab, nb_thresh=None):
    if nb_thresh is None:
        nb_thresh = []

    ncurves = len(scores)
    fpc = np.array([])
    tpc = np.array([])

    for n in range(ncurves):
        fp, tp, _, _ = myroc_n(scores[n], lab, nb_thresh)
        fp, tp, _, _ = rocch(fp, tp)

        tpc = np.concatenate((tpc, tp))
        fpc = np.concatenate((fpc, fp))

    fpc, tpc, auch, _ = rocch(fpc, tpc)
    return fpc, tpc, auch
