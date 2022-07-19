import glob
import numpy as np
import pandas as pd
from BBC_Algorithms.auc_bbc_algo import auc_bbc
from Tables.splitTable import split_table

path = "../PROMIS/CK/*.csv"
all_predict_proba = np.load('../Results_Dataframes/CK_Results/CK_all_six_models_predict_proba.npy', allow_pickle=True)
random_seed = 42
bbc_list = []
bbc_fpr_list = []
bbc_tpr_list = []
for count, fname in enumerate(glob.glob(path)):
    soft_detectors = all_predict_proba[count]
    ab = auc_bbc(split_table(fname, random_seed)[3], soft_detectors, 12)
    bbc_list.append(ab[0])
    bbc_fpr_list.append(ab[1])
    bbc_tpr_list.append(ab[2])
    print(count, bbc_list)
    # if count == 3:
    #     break

np.save('bbc_roc_fpr', np.array(bbc_fpr_list, dtype=object))
np.save('bbc_roc_tpr', np.array(bbc_tpr_list, dtype=object))
columns = ['BBC']
df = pd.DataFrame(bbc_list, columns=columns)

# df.to_csv('AUC_Table_BBC.csv')






