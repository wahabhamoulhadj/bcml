import numpy as np
from matplotlib import pyplot as plt

from BBC_Algorithms.boolean_generator_2array import bf_output
from BBC_Algorithms.rocch import rocch
from BBC_Algorithms.two_d_array_to_fpr_tpr import resp2pts


def bcvtt(s1, t1, s2, t2, lab, fun=None):
    if fun is None:
        fun = np.array(range(10), dtype=object)

    # if 1000000007 in t1:
    #   pass
    # else:
    #   t1 = np.append(t1, 1000000007)

    # if 1000000007 in t2:
    #   pass
    # else:
    #   t2 = np.append(t2, 1000000007)

    l = len(lab)
    l1 = len(t1)
    l2 = len(t2)

    FP = np.array([])
    TP = np.array([])
    BF = np.array([])
    TH1 = np.array([])
    TH2 = np.array([])
    RS = np.empty((0, len(lab)), int)

    for b in fun:
        rs = []
        for j in t1:
            r_1 = s1 > j
            for k in t2:
                r_2 = (s2 > k)
                r_12 = bf_output(b, r_1, r_2)
                rs.append(r_12)

        fpr_list, tpr_list = resp2pts(lab, rs)
        fpc, tpc, auch, ix = rocch(fpr_list, tpr_list)

        ix = np.array(ix)
        rs = np.array(rs)
        fpr_list = np.array(fpr_list)
        tpr_list = np.array(tpr_list)

        it1 = np.floor(ix / l2)
        it1 = it1.astype(int)
        it2 = ix % l2

        # TH1.append(t1[it1])
        TH1 = np.concatenate((TH1, t1[it1]))
        TH2 = np.concatenate((TH2, t2[it2]))

        # TH2.append(t2[it2])
        # RS.append(rs[ix])
        RS = np.concatenate((RS, rs[ix]), axis=0)
        BF = np.concatenate((BF, np.ones(len(ix)) * b))
        FP = np.concatenate((FP, fpr_list[ix]))
        TP = np.concatenate((TP, tpr_list[ix]))

    # TH1 = np.array(TH1)
    # TH2 = np.array(TH2)
    # BF = np.array(BF)
    # RS = np.array(RS)
    # FP = np.array(FP)
    # TP = np.array(TP)

    FP, TP, AUCH, IX = rocch(FP, TP)
    RS = RS[IX]
    TH1 = TH1[IX]
    TH2 = TH2[IX]
    BF = BF[IX]
    return RS, TH1, TH2, BF, FP, TP, AUCH

# rs, TH1, TH2, BF, FP, TP, AUCH = bcvtt(soft_detectors[0], thresholds[0], soft_detectors[1], thresholds[1], original)

# print(rs)
# print(TH1)
#
# print(TH2)
# print(BF)
# print(FP)
# print(TP)
# print(AUCH)
# plt.plot(FP, TP, marker = '+')
#
# plt.show()
