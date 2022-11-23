import glob
import numpy as np
import pandas as pd

from IBC_Algorithm.IBC_Pseudo_ALgo import IBC_Pseudo_Algo
from Tables.splitTable import split_table

path = "../../PROMIS/CK_NET_PROC/*.csv"
ibc_list = []
ibc_fpr = []
ibc_tpr = []
all_predict_proba = np.load('CK_NET_PROC_all_six_models_predict_proba.npy', allow_pickle=True)

for count, fname in enumerate(glob.glob(path)):
    soft_detectors = all_predict_proba[count]
    IBCvr = IBC_Pseudo_Algo(split_table(fname, 42)[3], soft_detectors, 12)
    ibc_list.append(IBCvr[0][2])
    ibc_fpr.append(IBCvr[0][0])
    ibc_tpr.append(IBCvr[0][1])

    print('-----------------------')
    print(count)
    print(IBCvr[0][2])

print(ibc_fpr)
print(ibc_tpr)
columns = ['IBC']
np.save('CK_NET_PROC_ibc_roc_fpr', np.array(ibc_fpr, dtype=object))
np.save('CK_NET_PROC_ibc_roc_tpr', np.array(ibc_tpr, dtype=object))
df = pd.DataFrame(ibc_list, columns=columns)
print(df)

df.to_csv('CK_NET_PROC_AUC_Table_IBC.csv')