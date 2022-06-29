import glob
import numpy as np
import pandas as pd
from IBC_Algorithm.IBC_ibcvr import ibcvr
from Tables.splitTable import split_table

path = "../PROMIS/CK/*.csv"
ibc_list = []
ibc_fpr = []
ibc_tpr = []
all_predict_proba = np.load('../Tables/all_six_models_predict_proba.npy', allow_pickle=True)
for count, fname in enumerate(glob.glob(path)):
    soft_detectors = all_predict_proba[count]
    IBCvr = ibcvr(soft_detectors, split_table(fname, 42)[3], 12)
    ibc_list.append(IBCvr[2])
    ibc_fpr.append(IBCvr[0])
    ibc_tpr.append(IBCvr[1])
    print(count, ibc_list)

columns = ['IBC']
np.save('ibc_roc_fpr', np.array(ibc_fpr, dtype=object))
np.save('ibc_roc_tpr', np.array(ibc_tpr, dtype=object))
df = pd.DataFrame(ibc_list, columns=columns)

df.to_csv('AUC_Table_IBC.csv')
