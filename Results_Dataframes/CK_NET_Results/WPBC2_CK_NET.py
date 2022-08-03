"""
BBC Algorithm training for a dataset
"""

# Import necessary files
import glob  #
import numpy as np
import pandas as pd
from BBC_Algorithms.BBC_Algorithm import BBC_Algo  # BBC Algorithm
from Tables.splitTable import split_table  # Splits a dataset to X_train, X_test, Y_train, Y_test,
# Y_test is our Ground Truth
from WPBC2_Algorithms.WPBC2_Algorithm import phase_1

path = "../../PROMIS/CK_NET/*.csv"
all_predict_proba = np.load('CK_NET_all_six_models_predict_proba.npy', allow_pickle=True)
# print(all_predict_proba)
random_seed = 42
wpbc2_list = []
wpbc2_fpr_list = []
wpbc2_tpr_list = []

for count, fname in enumerate(glob.glob(path)):  # Iterate through every dataset in CK_NET
    soft_detectors = all_predict_proba[count]
    # print(split_table(fname, random_seed)[3])
    # print(soft_detectors)
    ab = phase_1(split_table(fname, random_seed)[3], soft_detectors, 0.5, 0.5, 12)

    wpbc2_list.append(ab[0])
    wpbc2_fpr_list.append(ab[1])
    wpbc2_tpr_list.append(ab[2])
    print(count, wpbc2_list)
    # if count == 3:
    #     break

np.save('CK_NET_wpbc2_roc_fpr', np.array(wpbc2_fpr_list, dtype=object))
np.save('CK_NET_wpbc2_roc_tpr', np.array(wpbc2_tpr_list, dtype=object))
columns = ['WPBC2']
df = pd.DataFrame(wpbc2_list, columns=columns)
print(df)
df.to_csv('CK_NET_WPBC2_AUC_Table.csv')
