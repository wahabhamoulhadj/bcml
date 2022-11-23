"""
PBC Algorithm training for a dataset
"""

# Import necessary files
import glob  #
import numpy as np
import pandas as pd
from BBC_Algorithms.BBC_Algorithm import BBC_Algo  # BBC Algorithm
from PBC_Algorithms.PBC_Algoithm import pbc_Algo
from Tables.splitTable import split_table  # Splits a dataset to X_train, X_test, Y_train, Y_test,
# Y_test is our Ground Truth

path = "../../PROMIS/CK/*.csv"
all_predict_proba = np.load('CK_all_six_models_predict_proba.npy', allow_pickle=True)
print(all_predict_proba)
random_seed = 42
pbc_list = []
pbc_fpr_list = []
pbc_tpr_list = []
for count, fname in enumerate(glob.glob(path)): #Iterate through every dataset in CK
    soft_detectors = all_predict_proba[count]
    ab = pbc_Algo(split_table(fname, random_seed)[3], soft_detectors, 0.5, 12)
    pbc_list.append(ab[0])
    pbc_fpr_list.append(ab[1])
    pbc_tpr_list.append(ab[2])
    print(count, pbc_list)
    # if count == 3:
    #     break

np.save('CK_pbc_roc_fpr', np.array(pbc_fpr_list, dtype=object))
np.save('CK_pbc_roc_tpr', np.array(pbc_tpr_list, dtype=object))
columns = ['PBC']
df = pd.DataFrame(pbc_list, columns=columns)
print(df)
df.to_csv('CK_PBC_AUC_Table.csv')
