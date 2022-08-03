import glob

import numpy as np
import pandas as pd
from Tables.splitTable import split_table

from WPIBC_Algorithm.wpibc_algorithm import phase_3

path = "../../PROMIS/NET/*.csv"
WPibc_fpr = []
WPibc_list = []
WPibc_tpr = []
all_predict_proba = np.load('NET_all_six_models_predict_proba.npy', allow_pickle=True)
for count, fname in enumerate(glob.glob(path)):
    soft_detectors = all_predict_proba[count]
    WPIBCvr = phase_3(split_table(fname, 42)[3], soft_detectors, 12)
    WPibc_list.append(WPIBCvr[0][2])
    WPibc_fpr.append(WPIBCvr[0][0])
    WPibc_tpr.append(WPIBCvr[0][1])
    print(count, WPIBCvr[0][2])

    # print(count, wpibc_auc_list)
# print(wpibc_auc_list)

columns = ['WPIBC']
np.save('NET_wpibc_roc_fpr', np.array(WPibc_fpr, dtype=object))
np.save('NET_wpibc_roc_tpr', np.array(WPibc_tpr, dtype=object))
df = pd.DataFrame(WPibc_list, columns=columns)

df.to_csv('NET_AUC_Table_WPIBC.csv')