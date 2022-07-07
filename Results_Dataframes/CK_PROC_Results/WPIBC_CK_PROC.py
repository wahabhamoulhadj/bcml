import glob

import numpy as np
import pandas as pd

from IBC_Algorithm.IBC_ibcvr import ibcvr
from Tables.splitTable import split_table
from WPIBC_Algorithm.wtPrune import wt_prune

path = "../../PROMIS/CK_PROC/*.csv"
WPibc_fpr = []
WPibc_list = []
WPibc_tpr = []
all_predict_proba = np.load('CK_PROC_all_six_models_predict_proba.npy', allow_pickle=True)
for count, fname in enumerate(glob.glob(path)):
    soft_detectors = all_predict_proba[count]
    WPIBCvr = ibcvr(soft_detectors[wt_prune(soft_detectors, split_table(fname, 42)[3], 14, 0.8)], split_table(fname, 42)[3],
                    12)
    WPibc_list.append(WPIBCvr[2])
    WPibc_fpr.append(WPIBCvr[0])
    WPibc_tpr.append(WPIBCvr[1])
    print(count, WPibc_list)

    # print(count, wpibc_auc_list)
# print(wpibc_auc_list)

columns = ['WPIBC']
np.save('CK_PROC_wpibc_roc_fpr', np.array(WPibc_fpr, dtype=object))
np.save('CK_PROC_wpibc_roc_tpr', np.array(WPibc_tpr, dtype=object))
df = pd.DataFrame(WPibc_list, columns=columns)

df.to_csv('CK_PROC_AUC_Table_WPIBC.csv')
