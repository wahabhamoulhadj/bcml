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

path = "../../PROMIS/CK_NET_PROC/*.csv"
all_predict_proba = np.load('CK_NET_PROC_all_six_models_predict_proba.npy', allow_pickle=True)
print(all_predict_proba)
random_seed = 42
bbc_list = []
bbc_fpr_list = []
bbc_tpr_list = []
for count, fname in enumerate(glob.glob(path)): #Iterate through every dataset in CK_NET
    soft_detectors = all_predict_proba[count]
    ab = BBC_Algo(split_table(fname, random_seed)[3], soft_detectors, 12)
    bbc_list.append(ab[0])
    bbc_fpr_list.append(ab[1])
    bbc_tpr_list.append(ab[2])
    print(count, bbc_list)
    # if count == 3:
    #     break

np.save('CK_NET_PROC_bbc_roc_fpr', np.array(bbc_fpr_list, dtype=object))
np.save('CK_NET_PROC_bbc_roc_tpr', np.array(bbc_tpr_list, dtype=object))
columns = ['BBC']
df = pd.DataFrame(bbc_list, columns=columns)
print(df)
df.to_csv('CK_NET_PROC_BBC_AUC_Table.csv')
