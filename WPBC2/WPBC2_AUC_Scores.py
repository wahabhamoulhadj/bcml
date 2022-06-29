import glob

import numpy as np
import pandas as pd

from BBC_Algorithms.auc_bbc_algo import auc_bbc
from IBC_Algorithm.IBC_ibcvr import ibcvr
from Tables.splitTable import split_table
from WPIBC_Algorithm.wtPrune import wt_prune

path = "../PROMIS/CK/*.csv"
wpbc2_auc_list = []
random_seed = 42
all_predict_proba = np.load('../Tables/all_six_models_predict_proba.npy', allow_pickle=True)
for count, fname in enumerate(glob.glob(path)):
    soft_detectors = all_predict_proba[count]
    wpbc2_auc_list.append(auc_bbc(split_table(fname, random_seed)[3],
                                  soft_detectors[wt_prune(soft_detectors, split_table(fname, 42)[3], 14, 0.8)],
                                  12))
#     print(count, wpbc2_auc_list)
# print(wpbc2_auc_list)

columns = ['WPBC2']
df = pd.DataFrame(wpbc2_auc_list, columns=columns)

df.to_csv('AUC_Table_WPBC2.csv')
