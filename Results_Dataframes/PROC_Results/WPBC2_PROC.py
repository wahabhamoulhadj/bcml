import glob

import numpy as np
import pandas as pd

from BBC_Algorithms.auc_bbc_algo import auc_bbc
from IBC_Algorithm.IBC_ibcvr import ibcvr
from Tables.splitTable import split_table
from WPIBC_Algorithm.wtPrune import wt_prune

path = "../../PROMIS/PROC/*.csv"
wpbc2_auc_list = []
wpbc2_list = []
wpbc2_fpr_list = []
wpbc2_tpr_list = []
random_seed = 42
all_predict_proba = np.load('PROC_all_six_models_predict_proba.npy', allow_pickle=True)
for count, fname in enumerate(glob.glob(path)):
    soft_detectors = all_predict_proba[count]
    ab = auc_bbc(split_table(fname, random_seed)[3],
                 soft_detectors[wt_prune(soft_detectors, split_table(fname, 42)[3], 14, 0.8)],12)
    wpbc2_list.append(ab[0])
    wpbc2_fpr_list.append(ab[1])
    wpbc2_tpr_list.append(ab[2])
    print(count, ab[0])
# print(wpbc2_auc_list)

# columns = ['WPBC2']
# df = pd.DataFrame(wpbc2_auc_list, columns=columns)

# df.to_csv('PROC_WPBC2_AUC_Table.csv')

np.save('PROC_WPBC2_roc_fpr', np.array(wpbc2_fpr_list, dtype=object))
np.save('PROC_WPBC2_roc_tpr', np.array(wpbc2_tpr_list, dtype=object))
columns = ['WPBC2']
df = pd.DataFrame(wpbc2_list, columns=columns)
print(df)
df.to_csv('PROC_WPBC2_AUC_Table.csv')