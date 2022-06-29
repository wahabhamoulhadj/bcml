import numpy as np

from WPIBC_Algorithm.rocBug import roc_bug

import glob

import numpy as np

from WPIBC_Algorithm.PBC_contngncy import contngncy
from WPIBC_Algorithm.PBC_wtkappa import wtkappa
from WPIBC_Algorithm.rocBug import roc_bug

from scipy.stats.mstats import mquantiles

from Tables.splitTable import split_table

def pscds(val_scrs, val_lab, nb_thresh):
    no_sd = len(val_scrs)
    bsd = np.zeros((no_sd, 3))

    for i in range(no_sd):
        _, _, au, _ = roc_bug(val_scrs[i], val_lab, nb_thresh)
        bsd[i, 0] = i
        bsd[i, 1] = au
        bsd[i, 2] = val_scrs[i]


    b = []
    k = len(bsd)
    while k:
        sb = np.where(bsd[:, 1] == np.max(bsd[:, 1]))[0][0]
        b.append(bsd[sb])
        nb = nb_thresh
        bsd = np.delete(bsd, sb, 0)
        k = len(bsd)

        for l in range(k):
            lavls = mquantiles(bsd[l,3], np.linspace(0, 1, num=nb_thresh))
            lavls = np.unique(np.array(lavls))
            lavls = -np.sort(-lavls)
            tblContngncy = contngncy(scr1, scr2, lavls)
            kpp = wtkappa(tblContngncy, wt1)




    S = np.empty([])
    # return S
    k = no_sd
    hm_au = bsd

    kp_all = np.zeros((no_sd, 1))

    I = np.where(hm_au[:, 1] == np.max(hm_au[:, 1]))
    kp_all[I] = 1