import glob
import numpy as np
import pandas as pd
from IBC_Algorithm.IBC_ibcvr import ibcvr
from Tables.splitTable import split_table

path = "../../PROMIS/NET_PROC/*.csv"
ibc_list = []
ibc_fpr = []
ibc_tpr = []
all_predict_proba = np.load('NET_PROC_all_six_models_predict_proba.npy', allow_pickle=True)
for count, fname in enumerate(glob.glob(path)):
    soft_detectors = all_predict_proba[count]
    IBCvr = ibcvr(soft_detectors, split_table(fname, 42)[3], 12)
    ibc_list.append(IBCvr[2])
    ibc_fpr.append(IBCvr[0])
    ibc_tpr.append(IBCvr[1])
    print(count, ibc_list)

columns = ['IBC']
np.save('NET_PROC_ibc_roc_fpr', np.array(ibc_fpr, dtype=object))
np.save('NET_PROC_ibc_roc_tpr', np.array(ibc_tpr, dtype=object))
df = pd.DataFrame(ibc_list, columns=columns)

df.to_csv('NET_PROC_AUC_Table_IBC.csv')
