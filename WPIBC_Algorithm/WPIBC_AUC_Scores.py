import glob

import numpy as np
import pandas as pd

from IBC_Algorithm.IBC_ibcvr import ibcvr
from Tables.splitTable import split_table
from WPIBC_Algorithm.wtPrune import wt_prune

path = "../PROMIS/CK/*.csv"
wpibc_auc_list = []
all_predict_proba = np.load('../Tables/all_six_models_predict_proba.npy', allow_pickle=True)
for count, fname in enumerate(glob.glob(path)):
    soft_detectors = all_predict_proba[count]
    wpibc_auc_list.append(
        ibcvr(soft_detectors[wt_prune(soft_detectors, split_table(fname, 42)[3], 14, 0.8)], split_table(fname, 42)[3],
              12)[2])
    # print(count, wpibc_auc_list)
# print(wpibc_auc_list)

columns = ['WPIBC']
df = pd.DataFrame(wpibc_auc_list, columns=columns)

df.to_csv('AUC_Table_WPIBC.csv')
