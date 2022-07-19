import glob
import numpy as np
from WPIBC_Algorithm.PBC_contngncy import contngncy
from WPIBC_Algorithm.PBC_wtkappa import wtkappa
from WPIBC_Algorithm.rocBug import roc_bug
from scipy.stats.mstats import mquantiles
from Tables.splitTable import split_table
np.seterr(invalid='ignore')

def wt_prune(val_scrs, val_lab, nb_thresh, kp_th):
    no_sd = len(val_scrs)
    bsd = np.zeros((no_sd, 2))

    for i in range(no_sd):
        _, _, au, _ = roc_bug(val_scrs[i], val_lab, nb_thresh)
        bsd[i, 0] = i
        bsd[i, 1] = au

    S = np.empty([])
    # return S
    k = no_sd
    hm_au = bsd

    kp_all = np.zeros((no_sd, 1))

    I = np.where(hm_au[:, 1] == np.max(hm_au[:, 1]))
    kp_all[I] = 1
    flag = 1

    # print(val_scrs[hm_au[I, 0]])
    while k > 1:
        I = np.where(hm_au[:, 1] == np.max(hm_au[:, 1]))[0][0]
        bd = hm_au[I, 0].astype(int)
        # bd = np.array(bd, dtype=int)
        # print(bd)

        scr1 = val_scrs[bd]
        # print(scr1)

        left_h = np.array([], dtype=int)

        for i in range(k):
            hm = hm_au[i, 0].astype(int)
            # print(hm)
            # print(bd)
            if hm != bd:
                # print(hm)
                scr2 = val_scrs[hm]

                lavls = mquantiles(scr2, np.linspace(0, 1, num=nb_thresh))
                lavls = np.unique(np.array(lavls))
                lavls = -np.sort(-lavls)
                sz_l = len(lavls)

                wt1 = np.zeros((sz_l, sz_l))

                for m in range(sz_l):
                    for n in range(sz_l):
                        wt1[m, n] = 1 - (abs(m - n) / (sz_l - 1))
                # print(np.shape(scr1))
                # print(np.shape(scr2))
                # print(np.shape(lavls))

                tblContngncy = contngncy(scr1, scr2, lavls)
                kpp = wtkappa(tblContngncy, wt1)
                if flag:
                    kp_all[i] = kpp

                if kpp > kp_th:
                    continue
                else:
                    left_h = np.append(left_h, i)

        flag = 0
        # print(left_h)
        hm_au = hm_au[left_h, :]
        k = len(hm_au[:, 0])
        if k != 1:
            # print(S)
            # print(bd)
            S = np.append(S, bd)
        else:
            S = np.append(S, bd)
            S = np.append(S, hm_au[0, 0])
            break
    # print(S)
    # I = np.argsort(kp_all)
    # if S != bsd[I[0], 0]:
    #     S = np.append(S, bsd[I[0], 0])
    #
    # I = np.where(hm_au[:, 1] == np.min(hm_au[:, 1]))
    #
    # if S != bsd[I[0], 0]:
    #     S = np.append(S, bsd[I[0], 0])

    return S[1:].astype(int)


# path = "../PROMIS/CK/*.csv"
# wpibc_list = []
# all_predict_proba = np.load('../Tables/CK_all_six_models_predict_proba.npy', allow_pickle=True)
# for count, fname in enumerate(glob.glob(path)):
#     soft_detectors = all_predict_proba[count]
#     wpibc_list.append(wt_prune(soft_detectors, split_table(fname, 42)[3], 14, 0.8))
#     # print(count)
# print(wpibc_list)
